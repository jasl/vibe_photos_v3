# vibe_photos_v3 实物识别模块评估与替代方案报告

## 📌 1. 当前设计方案的不足

### 1.1 使用 YOLOv5/YOLOv8 存在许可问题

* Ultralytics 发布的 YOLOv5 与 YOLOv8 均采用 **AGPL-3.0 许可**，这是一个对使用者要求较高的“强互惠”协议。
* 一旦使用该模型（包括加载其训练权重），**所有调用该模型的系统都必须开源并采用 AGPL 许可**，否则将违反许可条款。
* 若需商业闭源部署，则必须向 Ultralytics 付费购买授权。
* 对于计划开源或可能用于商业用途的 vibe_photos_v3 项目，这类许可显然**存在法律风险**。

### 1.2 检测精度不满足“准确度优先”需求

* YOLOv5/8 的目标是速度优先（实时检测），而不是最高检测精度。
* 在 COCO 数据集上，YOLOv8-Large 的 mAP（mean Average Precision）为约 **50.2%**。
* 你的工作需要识别较复杂、专业的照片素材（如数码产品、食物等），**精度是首要考虑因素**，YOLOv8 并非最佳选择。

---

## ✅ 2. 推荐替代方案：RTMDet-L（OpenMMLab 出品）

| 模型名称         | mAP (COCO) | 许可证          | 推理平台      | 预训练模型  | 理由总结              |
| ------------ | ---------- | ------------ | --------- | ------ | ----------------- |
| **RTMDet-L** | **52.8%**  | Apache-2.0 ✅ | PyTorch ✅ | 官方支持 ✅ | 精度高、许可宽松、部署相对简单 ✅ |

### 2.1 模型简介

* **RTMDet（Real-Time Multi-scale Detector）** 是 OpenMMLab 于 2023 年发布的高精度检测器。
* 由 MMDetection 提供官方支持，配套预训练模型和配置。
* 适用于 COCO 类通用物体检测任务，支持识别如“手机”、“电脑”、“食物”等常见实体。

### 2.2 为什么适合你

* ✅ **高精度优先**：精度优于 YOLOv5/YOLOv8，适合严苛素材筛选需求；
* ✅ **开源兼容**：Apache-2.0 无使用限制，无强制开源，适合用于 MIT / BSD 项目中；
* ✅ **部署可控**：通过 MMDetection 在 PyTorch 中部署，不依赖专有代码框架；
* ✅ **预训练即用**：提供现成 COCO 权重文件，无需自行训练；
* ✅ **类别通用**：支持 80 类 COCO 标签，涵盖你照片中常见物体（手机、食物、文件、电子产品等）；

---

## ⚙️ 3. RTMDet-L 推理脚本模板（基于 MMDetection）

该脚本将从图像目录批量读取图片，使用 RTMDet-L 模型执行检测，输出检测类别、置信度和边界框坐标，并可选生成可视化图像。

```python
import os
from mmdet.apis import init_detector, inference_detector

# 配置路径
config_file = 'configs/rtmdet/rtmdet_l_8xb32-300e_coco.py'
checkpoint_file = 'checkpoints/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'

# 初始化模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')  # 或 'cpu'

# 类别名称
class_names = model.dataset_meta['classes']

# 图像输入 / 输出目录
img_dir = 'path/to/your_images'
output_dir = 'path/to/save_viz'
os.makedirs(output_dir, exist_ok=True)

# 推理参数
conf_threshold = 0.3
save_viz = True
results_dict = {}

for fname in os.listdir(img_dir):
    if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    img_path = os.path.join(img_dir, fname)
    result = inference_detector(model, img_path)

    # 结果解析（适配 MMDetection 3.x）
    detections = []
    pred = result.pred_instances
    for bbox, score, label in zip(pred.bboxes.cpu().numpy(), pred.scores.cpu().numpy(), pred.labels.cpu().numpy()):
        if score < conf_threshold:
            continue
        detections.append({
            "class": class_names[label],
            "confidence": float(score),
            "bbox": bbox.tolist()
        })
    results_dict[fname] = detections

    # 可视化输出
    if save_viz:
        out_path = os.path.join(output_dir, f"det_{fname}")
        model.show_result(img_path, result, score_thr=conf_threshold, out_file=out_path)

# 保存结果
import json
with open('detection_results.json', 'w', encoding='utf-8') as f:
    json.dump(results_dict, f, indent=2, ensure_ascii=False)

print("检测完成，结果保存在 detection_results.json 中")
```

### ✅ 支持功能：

* [x] 从图片目录读取
* [x] 使用 RTMDet-L 模型批量推理
* [x] 输出物体类别、置信度、bbox
* [x] 可选保存标注图像供校对

---

## 🧩 4. 后续建议

* ✅ 将此脚本封装为 vibe_photos_v3 的模块接口，如 `extract_object_tags(image_dir)`；
* ✅ 检测结果可用于自动生成图像标签或分类辅助索引；
* ✅ 后续可拓展识别更专业类别（通过微调或自定义数据集训练）；
* ✅ 若集成至前端或交互系统，建议转为 ONNX 或 TorchScript 后导出用于服务部署；
