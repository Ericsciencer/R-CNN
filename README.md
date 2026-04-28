# R‑CNN (Regions with CNN features)

### 选择语言 | Language
[中文简介](#简介) | [English](#introduction)

### 结果 | Result
<img width="1036" height="673" alt="class_ap_bar" src="https://github.com/user-attachments/assets/d9edb327-035d-4591-9447-e3a149c48307" />

<img width="830" height="817" alt="pr_curve" src="https://github.com/user-attachments/assets/67ef1839-64fe-432c-ba46-26c3e3197cf0" />


<img width="1016" height="586" alt="iou_dist" src="https://github.com/user-attachments/assets/01e80a86-18d3-42ea-99bf-6cbd4d15bb54" />

<img width="775" height="673" alt="prf1_bar" src="https://github.com/user-attachments/assets/a2b06a6c-fed3-42d4-b3e2-76c06980210d" />

我们对于结果发现，他其实存在假阳性，这需要我们进行调参各种处理，虽然R-CNN是目标检测开篇之作，但是也几乎不再用的东西，后面做了不少改进，这里只复现跑通学习思想，学习网络结构，不再调参。

---

## 简介
R‑CNN（Regions with CNN features）由 Ross Girshick 等人于 2014 年提出，是目标检测领域**首次成功将深度卷积神经网络与区域提议相结合的里程碑工作**。在 R‑CNN 之前，目标检测主要依赖手工特征（如 HOG、SIFT）搭配滑窗或部件模型，检测精度受限且难以泛化。R‑CNN 独创性地引入“区域提议 + CNN 特征提取 + SVM 分类”的级联框架，并补充**边界框回归**进行空间精修，在 PASCAL VOC 2012 数据集上将 mAP 一举提升至 53.3%，较传统方法提升超过 30%。该方法不仅证明了预训练 CNN 可以迁移为强大的目标检测特征提取器，更开创了**两阶段检测器**（Two‑Stage Detector）的先河，直接启发 Fast R‑CNN、Faster R‑CNN 等后续经典工作，是深度学习目标检测领域的奠基之作。

<img width="500" height="297" alt="test_detection_output" src="https://github.com/user-attachments/assets/cde67305-e167-4d87-8bf3-fc7a9d17ad2e" />

## 架构
本次复现严格遵循原始 R‑CNN 的四阶段设计，整体分为「候选区域生成」「CNN 特征提取」「SVM 分类与边框回归训练」和「推理后处理」四大模块，并使用预训练 AlexNet 作为特征提取骨干网络。为适应小规模复现需求，在数据集和部分超参数上进行了适当调整，但核心架构完全保留。

- **候选区域生成模块**：采用**选择性搜索算法**，基于图像的色彩、纹理、尺寸等底层特征进行超像素合并，为每张图像生成约 2000 个类别无关的候选区域。这些候选区域具有多尺度、多宽高比的特点，能够以极高的召回率覆盖真实物体位置。
- **CNN 特征提取模块**：每个候选区域被各向同性缩放至 227×227 固定尺寸后，送入**预训练的 AlexNet** 模型。移除网络最后的分类层，抽取 4096 维的 fc7 特征向量作为候选区域的视觉表示。该模块在训练与推理中权重冻结，不参与梯度更新。
- **SVM 分类与边界框回归模块**（离线训练）：  
  - **训练阶段**：对所有候选框提取特征后，根据与真实标注框的 IoU 划分正负样本（正样本阈值 > 0.5，负样本阈值 < 0.3）。为每个目标类别单独训练一个线性 SVM 二分类器，用于区分“该类物体”与“背景”。  
  - **边界框回归**：仅使用正样本训练一个线性回归器，学习从候选框坐标到真实框坐标的尺度不变变换参数 (dx, dy, dw, dh)，实现定位精修。
- **推理与后处理模块**：推理时，对测试图像生成的每个候选框依次提取 CNN 特征、经各类 SVM 打分、取最高分且超过置信度阈值的类别作为预测结果，再应用该类回归器修正框坐标，最后通过非极大值抑制去除重叠冗余框，输出最终检测结果。

R-CNN执行流程：
<img width="1260" height="191" alt="image" src="https://github.com/user-attachments/assets/5450235b-8eca-4bf4-9c90-6461e78d0275" />
<img width="765" height="290" alt="image" src="https://github.com/user-attachments/assets/6e67cf4b-2221-4097-9113-fe4f6f172901" />

<img width="1574" height="565" alt="image" src="https://github.com/user-attachments/assets/0fbd8a17-202d-4999-9b20-f537f1dd162f" />
<img width="1250" height="793" alt="image" src="https://github.com/user-attachments/assets/48c4ec13-0c25-4676-b0c1-32dbf023eb8f" />

候选框产生：
<img width="1489" height="405" alt="image" src="https://github.com/user-attachments/assets/8d4aec4c-9ad3-4f8e-a896-7dffb0038767" />
<img width="1300" height="829" alt="image" src="https://github.com/user-attachments/assets/739278cd-93a6-4c8c-ace5-0c49a6230584" />

边界框回归：
<img width="906" height="809" alt="image" src="https://github.com/user-attachments/assets/2c752ef4-afee-40f9-9823-fd2f17435561" />

效率：
<img width="1168" height="855" alt="image" src="https://github.com/user-attachments/assets/63702495-f10e-41d6-be51-0885a63b9c7d" />

**注意**：为了便于快速复现与调试，本实现使用 **VOC 2007 子集**（仅 bird, cat, dog 三类）替换完整数据集，并将选择性搜索生成的候选框数量缩减至 1000 以内，大幅降低训练时间。但模型骨干、SVM 与回归训练逻辑、NMS 后处理等关键环节均严格复现原文设计。

## 数据集
我们使用从 PASCAL VOC 2007 中抽取的自定义子集，仅保留 **bird（鸟）、cat（猫）、dog（狗）** 三个类别。
- **数据来源**：完整 VOC 2007 train/val 集（约 450 MB）经筛选后，仅保留包含目标类别的图片及其标注，最终子集约包含 300‑500 张训练/验证图片。
- **标注格式**：PASCAL VOC XML 格式，每张图片对应一个 `.xml` 文件，包含目标类别名称和边界框坐标 (xmin, ymin, xmax, ymax)。
- **数据划分**：随机选择 80% 作为训练集，20% 作为验证集，用于模型训练与指标评估。

数据集官方地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2007/

---

## Introduction
R‑CNN (Regions with CNN features), proposed by Ross Girshick et al. in 2014, is a milestone work that **first successfully combined deep convolutional neural networks with region proposals** for object detection. Before R‑CNN, detection relied heavily on hand‑crafted features with sliding windows or part‑based models, resulting in limited accuracy. R‑CNN innovatively introduced a cascade framework of **region proposal + CNN feature extraction + SVM classification**, augmented with **bounding‑box regression** for spatial refinement. It achieved 53.3% mAP on PASCAL VOC 2012, a >30% improvement over traditional methods. This work proved that pre‑trained CNNs can serve as powerful feature extractors for detection, pioneering the **two‑stage detector** paradigm and directly inspiring Fast R‑CNN, Faster R‑CNN, and subsequent classics.
<img width="500" height="297" alt="test_detection_output" src="https://github.com/user-attachments/assets/cde67305-e167-4d87-8bf3-fc7a9d17ad2e" />

## Architecture
This reproduction strictly follows the original four‑stage design: region proposal, CNN feature extraction, SVM & bounding‑box regressor training, and inference post‑processing, using pre‑trained AlexNet as the backbone.

- **Region Proposal Module**: Uses **selective search** to generate ~2000 class‑agnostic candidate regions, achieving high recall via multi‑scale, multi‑aspect‑ratio coverage.
- **CNN Feature Extraction Module**: Each proposal is isotropically warped to 227×227 and passed through a **pre‑trained AlexNet** (fc7 layer, 4096‑d). Weights are frozen during training/inference.
- **SVM & Bounding‑Box Regressor Training** (offline):  
  - During training, proposals are labeled as positive (IoU > 0.5) or negative (IoU < 0.3). A linear SVM is trained per class to distinguish objects from background.  
  - A linear regressor is trained per class on positive samples only, predicting scale‑invariant coordinate transformations (dx, dy, dw, dh) from proposals to ground truth.
- **Inference & Post‑processing**: At test time, each proposal is scored by all class SVMs; the highest‑scoring class above a confidence threshold is selected. The corresponding regressor refines the box, and **non‑maximum suppression** removes redundant detections.

R-CNN execution flow:
<img width="1260" height="191" alt="image" src="https://github.com/user-attachments/assets/5450235b-8eca-4bf4-9c90-6461e78d0275" />
<img width="765" height="290" alt="image" src="https://github.com/user-attachments/assets/6e67cf4b-2221-4097-9113-fe4f6f172901" />

<img width="1574" height="565" alt="image" src="https://github.com/user-attachments/assets/0fbd8a17-202d-4999-9b20-f537f1dd162f" />
<img width="1250" height="793" alt="image" src="https://github.com/user-attachments/assets/48c4ec13-0c25-4676-b0c1-32dbf023eb8f" />

Selective Search：
<img width="1489" height="405" alt="image" src="https://github.com/user-attachments/assets/8d4aec4c-9ad3-4f8e-a896-7dffb0038767" />
<img width="1300" height="829" alt="image" src="https://github.com/user-attachments/assets/739278cd-93a6-4c8c-ace5-0c49a6230584" />

Bounding box regression:
<img width="906" height="809" alt="image" src="https://github.com/user-attachments/assets/2c752ef4-afee-40f9-9823-fd2f17435561" />

efficiency:
<img width="1168" height="855" alt="image" src="https://github.com/user-attachments/assets/63702495-f10e-41d6-be51-0885a63b9c7d" />

**Note**: For efficient reproduction, we adopt a **VOC 2007 subset** (bird, cat, dog only) and limit proposal count to ≤1000. Core components (backbone, SVM training, box regression, NMS) remain consistent with the original paper.

## Dataset
We use a custom subset extracted from PASCAL VOC 2007, containing only three classes: **bird, cat, dog**.
- **Source**: The full VOC 2007 train/val set was filtered to retain images with at least one target object. The subset contains 300‑500 images.
- **Annotation format**: PASCAL VOC XML with object class names and bounding box coordinates (xmin, ymin, xmax, ymax).
- **Split**: 80% for training, 20% for validation.

Official dataset page: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/

---
## 原文章 | Original article
Girshick R, Donahue J, Darrell T, et al. Rich feature hierarchies for accurate object detection and semantic segmentation[C]. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2014: 580‑587. arXiv preprint: arXiv:1311.2524.
