"""
R‑CNN 论文复现 —— 最终版
数据集：VOC 2007 子集 (bird, cat, dog)
特征网络：预训练 AlexNet
SVM + 边框回归 + 完整指标可视化
"""

import os, random, pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import alexnet, AlexNet_Weights
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

# 选择性搜索
import selectivesearch

# 处理 Colab 与本地显示
try:
    from google.colab.patches import cv2_imshow
    IN_COLAB = True
except:
    IN_COLAB = False
    cv2_imshow = cv2.imshow

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = (227, 227)
MAX_PROPOSALS = 1000
IOU_POS_THRESH = 0.5
IOU_NEG_THRESH = 0.3
NMS_THRESH = 0.3
CONF_THRESH = 0.6
EPS = 1e-8

CLASS_NAMES = ["background", "bird", "cat", "dog"]
NUM_CLASSES = len(CLASS_NAMES)

# 图像预处理
TRANSFORM = T.Compose([
    T.Resize(INPUT_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

SAVE_DIR = "rcnn_voc_tiny_result"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 工具函数 =====================
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1]) + EPS
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1]) + EPS
    union = area1 + area2 - inter
    return inter / union

def nms(boxes, scores, thresh):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1)*(y2-y1) + EPS
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        iou = w * h / areas[order[1:]]
        order = order[np.where(iou <= thresh)[0] + 1]
    return keep

def compute_pr_f1(tp, fp, fn):
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    return precision, recall, f1

def compute_ap(recalls, precisions):
    recalls = np.concatenate([[0.], recalls, [1.]])
    precisions = np.concatenate([[0.], precisions, [0.]])
    for i in range(len(precisions)-2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])
    i = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[i+1] - recalls[i]) * precisions[i+1])
    return ap

# ===================== 可视化函数 =====================
def plot_pr_curve(recall_list, precision_list):
    plt.figure(figsize=(6,6))
    plt.plot(recall_list, precision_list, c='#1f77b4', linewidth=2, label='PR Curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{SAVE_DIR}/pr_curve.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_iou_dist(iou_list):
    plt.figure(figsize=(8,4))
    sns.histplot(iou_list, bins=20, color='#2ca02c', alpha=0.7)
    plt.title("Detection IoU Distribution")
    plt.xlabel("IoU")
    plt.ylabel("Count")
    plt.savefig(f"{SAVE_DIR}/iou_dist.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_class_ap_bar(ap_dict):
    classes = list(ap_dict.keys())
    ap_values = list(ap_dict.values())
    plt.figure(figsize=(8,5))
    plt.bar(classes, ap_values, color='#ff7f0e', alpha=0.8)
    plt.ylim(0, 1.05)
    plt.title("Class-wise AP Bar Chart")
    plt.ylabel("AP Value")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(f"{SAVE_DIR}/class_ap_bar.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_prf1_bar(prec, rec, f1):
    metrics = ["Precision", "Recall", "F1-Score"]
    values = [prec, rec, f1]
    plt.figure(figsize=(6,5))
    plt.bar(metrics, values, color=['#1f77b4','#2ca02c','#d62728'], alpha=0.8)
    plt.ylim(0, 1.05)
    plt.title("Overall Detection Metrics")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(f"{SAVE_DIR}/prf1_bar.png", dpi=150, bbox_inches="tight")
    plt.show()

def vis_detection(img, boxes, cls_ids, scores):
    if img is None:
        print("No image to display")
        return
    img_copy = img.copy()
    color_map = {1: (0,255,0), 2: (255,0,0), 3: (0,0,255)}
    for box, cls, score in zip(boxes, cls_ids, scores):
        x1,y1,x2,y2 = map(int, box)
        color = color_map.get(cls, (0,255,0))
        cv2.rectangle(img_copy, (x1,y1), (x2,y2), color, 2)
        text = f"{CLASS_NAMES[cls]} {score:.2f}"
        cv2.putText(img_copy, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(f"{SAVE_DIR}/det_result.jpg", img_copy)
    cv2_imshow(img_copy)

# ===================== 候选框提取 =====================
def extract_proposals(img):
    if img is None or img.size == 0:
        return []
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 使用 selectivesearch 模块调用函数
    _, regions = selectivesearch.selective_search(img_rgb, scale=500, sigma=0.9, min_size=20)
    boxes = set()
    for reg in regions:
        x,y,w,h = reg["rect"]
        if w<20 or h<20: continue
        boxes.add((x,y,x+w,y+h))
    return list(boxes)[:MAX_PROPOSALS]

# ===================== VOC 2007 Tiny 数据集 =====================
class VOCTinyDataset:
    def __init__(self, root_dir="VOC2007_tiny", split="train"):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "JPEGImages")
        self.annot_dir = os.path.join(root_dir, "Annotations")
        self.split = split

        # 读取 train.txt 文件
        with open(os.path.join(root_dir, "ImageSets/Main/train.txt"), 'r') as f:
            all_ids = [line.strip() for line in f if line.strip()]

        # 简单划分 80% 训练，20% 验证
        np.random.seed(42)
        np.random.shuffle(all_ids)
        split_idx = int(0.8 * len(all_ids))
        if split == "train":
            self.img_ids = all_ids[:split_idx]
        else:
            self.img_ids = all_ids[split_idx:]

        print(f"VOC Tiny {split} set: {len(self.img_ids)} images")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_id + ".jpg")
        annot_path = os.path.join(self.annot_dir, img_id + ".xml")

        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self))

        h, w = img.shape[:2]
        gt_boxes, gt_classes = [], []

        # 解析 XML
        if not os.path.exists(annot_path):
            return self.__getitem__((idx + 1) % len(self))

        tree = ET.parse(annot_path)
        root = tree.getroot()
        class_to_id = {"bird": 1, "cat": 2, "dog": 3}

        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in class_to_id:
                continue
            cls_id = class_to_id[name]
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            if xmax - xmin < 1 or ymax - ymin < 1:
                continue
            gt_boxes.append([xmin, ymin, xmax, ymax])
            gt_classes.append(cls_id)

        if len(gt_boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))

        return img, gt_boxes, gt_classes

# ===================== RCNN 特征网络 =====================
class RCNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        alex = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.feature = nn.Sequential(*list(alex.features))
        self.fc = nn.Sequential(*list(alex.classifier)[:-1])   # 移除最后一层

    def forward(self, x):
        with torch.no_grad():
            feat = self.feature(x)
            feat = torch.flatten(feat, 1)
            feat = self.fc(feat)    # 4096 维特征
        return feat

# ===================== RCNN 核心类 =====================
class RCNN:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.backbone = RCNNBackbone().to(DEVICE)
        self.svms = [SVC(kernel="linear", probability=True) for _ in range(num_classes)]
        self.bbox_reg = [LinearRegression() for _ in range(num_classes)]
        self.all_iou = []
        self.precision_list = []
        self.recall_list = []
        self.fitted_classes = []

    def extract_roi_feat(self, img, box):
        x1,y1,x2,y2 = box
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(img.shape[1]-1, int(x2)), min(img.shape[0]-1, int(y2))
        if x2<=x1 or y2<=y1:
            return np.zeros(4096)

        crop = img[y1:y2, x1:x2]
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
        feat = self.backbone(tensor).cpu().numpy().squeeze()
        return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    def train_svm_bbox(self, train_data):
        print("=== 训练 SVM 和边框回归（标准 R‑CNN 流程）===")
        class_feats = {c: {"pos": [], "neg": []} for c in range(1, self.num_classes)}

        for img, gt_boxes, gt_classes in tqdm(train_data, desc="处理训练数据"):
            proposals = extract_proposals(img)
            if not proposals: continue

            for box in proposals:
                max_iou = 0
                matched_cls = 0
                matched_gt = None
                for gt_box, gt_cls in zip(gt_boxes, gt_classes):
                    iou = calculate_iou(box, gt_box)
                    if iou > max_iou:
                        max_iou = iou
                        matched_cls = gt_cls
                        matched_gt = gt_box

                feat = self.extract_roi_feat(img, box)
                if max_iou > IOU_POS_THRESH and matched_cls != 0:
                    class_feats[matched_cls]["pos"].append(feat)
                    # 边框回归目标
                    px, py = (box[0]+box[2])/2, (box[1]+box[3])/2
                    pw, ph = box[2]-box[0]+EPS, box[3]-box[1]+EPS
                    gx, gy, gw, gh = matched_gt[0], matched_gt[1], matched_gt[2]-matched_gt[0]+EPS, matched_gt[3]-matched_gt[1]+EPS
                    dx = (gx - px)/pw
                    dy = (gy - py)/ph
                    dw = np.log(gw/pw)
                    dh = np.log(gh/ph)
                    if not hasattr(self.bbox_reg[matched_cls], "feats"):
                        self.bbox_reg[matched_cls].feats = []
                        self.bbox_reg[matched_cls].deltas = []
                    self.bbox_reg[matched_cls].feats.append(feat)
                    self.bbox_reg[matched_cls].deltas.append([dx, dy, dw, dh])
                elif max_iou < IOU_NEG_THRESH:
                    for c in range(1, self.num_classes):
                        class_feats[c]["neg"].append(feat)

        # 训练 SVM
        print("=== 训练 SVM 分类器 ===")
        for c in range(1, self.num_classes):
            pos_feats = np.array(class_feats[c]["pos"])
            neg_feats = np.array(class_feats[c]["neg"])
            if len(pos_feats) == 0 or len(neg_feats) == 0:
                print(f"类别 {CLASS_NAMES[c]} 样本不足，跳过")
                continue
            if len(neg_feats) > len(pos_feats)*3:
                neg_feats = neg_feats[:len(pos_feats)*3]
            X = np.vstack([pos_feats, neg_feats])
            y = np.array([1]*len(pos_feats) + [0]*len(neg_feats))
            self.svms[c].fit(X, y)
            self.fitted_classes.append(c)
            print(f"类别 {CLASS_NAMES[c]} SVM 训练完成")

        # 训练边框回归器
        print("=== 训练边框回归器 ===")
        for c in range(1, self.num_classes):
            if hasattr(self.bbox_reg[c], "feats") and len(self.bbox_reg[c].feats) > 0:
                X = np.array(self.bbox_reg[c].feats)
                Y = np.array(self.bbox_reg[c].deltas)
                self.bbox_reg[c].fit(X, Y)
                print(f"类别 {CLASS_NAMES[c]} 回归器训练完成")

    def predict(self, img):
        proposals = extract_proposals(img)
        if not proposals:
            return [], [], []
        feats = np.array([self.extract_roi_feat(img, b) for b in proposals])
        final_boxes, final_cls, final_scores = [], [], []

        for idx, box in enumerate(proposals):
            feat = feats[idx:idx+1]
            best_sc, best_c = 0.0, 0
            for c in self.fitted_classes:
                sc = self.svms[c].predict_proba(feat)[0][1]
                if sc > best_sc:
                    best_sc, best_c = sc, c

            if best_sc < CONF_THRESH:
                continue

            # 边框回归修正
            if hasattr(self.bbox_reg[best_c], "coef_"):
                dx, dy, dw, dh = self.bbox_reg[best_c].predict(feat)[0]
                x1, y1, x2, y2 = box
                bw, bh = x2-x1+EPS, y2-y1+EPS
                nx1 = max(0, x1 + dx*bw)
                ny1 = max(0, y1 + dy*bh)
                nx2 = min(img.shape[1], x2 + dw*bw)
                ny2 = min(img.shape[0], y2 + dh*bh)
                final_boxes.append([nx1, ny1, nx2, ny2])
                final_cls.append(best_c)
                final_scores.append(best_sc)

        if len(final_boxes) > 0:
            boxes_arr = np.array(final_boxes)
            score_arr = np.array(final_scores)
            keep = nms(boxes_arr, score_arr, NMS_THRESH)
            return boxes_arr[keep], np.array(final_cls)[keep], score_arr[keep]
        return [], [], []

    def evaluation(self, val_data):
        all_tp, all_fp, all_fn = 0, 0, 0
        ap_dict = {}
        self.all_iou = []   # 清空历史记录
        eval_classes = [c for c in self.fitted_classes]

        for cls in eval_classes:
            cls_recall, cls_precision = [], []
            for img, gt_boxes, gt_classes in tqdm(val_data, desc=f"评估 {CLASS_NAMES[cls]}"):
                pred_boxes, pred_cls, pred_scores = self.predict(img)
                gt_cls_boxes = [box for box, c in zip(gt_boxes, gt_classes) if c == cls]

                tp, fp = 0, 0
                fn = len(gt_cls_boxes)
                matched = [False] * len(gt_cls_boxes)

                for pb, pc, ps in zip(pred_boxes, pred_cls, pred_scores):
                    if pc != cls:
                        fp += 1
                        continue
                    best_iou = 0
                    best_idx = -1
                    for i, gt_box in enumerate(gt_cls_boxes):
                        iou = calculate_iou(pb, gt_box)
                        if iou > best_iou and not matched[i]:
                            best_iou = iou
                            best_idx = i
                    if best_iou > IOU_POS_THRESH and best_idx != -1:
                        matched[best_idx] = True
                        tp += 1
                        fn -= 1
                        self.all_iou.append(best_iou)
                    else:
                        fp += 1

                p, r, _ = compute_pr_f1(tp, fp, fn)
                cls_precision.append(p)
                cls_recall.append(r)
                all_tp += tp
                all_fp += fp
                all_fn += fn

            ap_dict[CLASS_NAMES[cls]] = compute_ap(np.array(cls_recall), np.array(cls_precision))
            self.precision_list.append(np.mean(cls_precision))
            self.recall_list.append(np.mean(cls_recall))

        mean_prec, mean_recall, mean_f1 = compute_pr_f1(all_tp, all_fp, all_fn)
        mAP = np.mean(list(ap_dict.values()))

        print("\n" + "="*60)
        print("📌 R‑CNN 标准目标检测评价指标")
        print("="*60)
        print(f"Precision: {mean_prec:.4f}")
        print(f"Recall:    {mean_recall:.4f}")
        print(f"F1-Score:  {mean_f1:.4f}")
        print(f"mAP:       {mAP:.4f}")
        for name, ap in ap_dict.items():
            print(f"  {name:<8} AP: {ap:.4f}")
        print("="*60)

        # 可视化
        plot_pr_curve(self.recall_list, self.precision_list)
        if len(self.all_iou) > 0:
            plot_iou_dist(self.all_iou)
        else:
            print("No IoU data for distribution plot.")
        plot_class_ap_bar(ap_dict)
        plot_prf1_bar(mean_prec, mean_recall, mean_f1)

        return {"precision": mean_prec, "recall": mean_recall, "f1": mean_f1, "mAP": mAP}

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 确保 VOC2007_tiny 文件夹在当前目录
    # 如果没有，会提示错误
    if not os.path.exists("VOC2007_tiny"):
        print("请先下载并解压 VOC2007_tiny 数据集！")
        exit()

    train_dataset = VOCTinyDataset(root_dir="VOC2007_tiny", split="train")
    val_dataset   = VOCTinyDataset(root_dir="VOC2007_tiny", split="val")

    rcnn = RCNN(num_classes=NUM_CLASSES)
    rcnn.train_svm_bbox(train_dataset)

    if len(rcnn.fitted_classes) > 0:
        rcnn.evaluation(val_dataset)

        # 测试单张图片
        test_img_path = os.path.join("VOC2007_tiny", "JPEGImages", train_dataset.img_ids[0] + ".jpg")
        test_img = cv2.imread(test_img_path)
        if test_img is not None:
            det_boxes, det_cls, det_scores = rcnn.predict(test_img)
            vis_detection(test_img, det_boxes, det_cls, det_scores)
    else:
        print("没有成功训练任何类别，请检查数据集或调整超参数。")