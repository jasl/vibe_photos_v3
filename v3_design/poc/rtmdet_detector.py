#!/usr/bin/env python3
"""
RTMDet-L 检测器 POC - 展示高精度物体检测功能
基于OpenMMLab的MMDetection框架，Apache-2.0许可

使用示例：
python rtmdet_detector.py /path/to/image.jpg
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass
class RTMDetResult:
    """RTMDet检测结果数据类"""
    detections: List[Dict]  # 检测到的物体列表
    image_path: str
    inference_time: float
    model_info: Dict


class RTMDetDetector:
    """
    基于RTMDet-L的物体检测器
    使用MMDetection框架，提供高精度物体检测
    """
    
    # COCO数据集的80个类别
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # 适合自媒体创作者的重点类别
    PRIORITY_CLASSES = {
        '电子产品': ['laptop', 'cell phone', 'keyboard', 'mouse', 'tv', 'remote'],
        '美食': ['pizza', 'donut', 'cake', 'sandwich', 'hot dog', 'banana', 'apple', 'orange'],
        '生活用品': ['bottle', 'cup', 'bowl', 'fork', 'knife', 'spoon', 'book', 'clock'],
        '家具': ['chair', 'couch', 'bed', 'dining table', 'potted plant'],
        '交通工具': ['car', 'bicycle', 'motorcycle', 'bus', 'train', 'truck', 'airplane', 'boat']
    }
    
    def __init__(self, model_config: Optional[str] = None, device: str = 'cuda:0'):
        """
        初始化RTMDet检测器
        
        Args:
            model_config: 模型配置文件路径
            device: 运行设备 ('cuda:0' 或 'cpu')
        """
        self.device = device
        self.model_config = model_config or 'rtmdet_l_8xb32-300e_coco.py'
        self.checkpoint = 'rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth'
        
        try:
            from mmdet.apis import init_detector, inference_detector
            self.init_detector = init_detector
            self.inference_detector = inference_detector
            self._init_model()
        except ImportError:
            print("请先安装MMDetection: pip install mmdet==3.3.0 mmengine==0.10.7 mmcv==2.2.0")
            self.model = None
    
    def _init_model(self):
        """初始化MMDetection模型"""
        # 配置文件和权重路径
        config_file = f'configs/rtmdet/{self.model_config}'
        checkpoint_file = f'checkpoints/{self.checkpoint}'
        
        # 如果文件不存在，提供下载指引
        if not Path(config_file).exists():
            print(f"配置文件未找到: {config_file}")
            print("请从MMDetection官方仓库下载配置文件")
            print("https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet")
            self.model = None
            return
        
        if not Path(checkpoint_file).exists():
            print(f"模型权重未找到: {checkpoint_file}")
            print("请下载RTMDet-L预训练权重:")
            print("wget https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth")
            self.model = None
            return
        
        # 初始化模型
        self.model = self.init_detector(config_file, checkpoint_file, device=self.device)
        print(f"RTMDet-L模型加载成功 (设备: {self.device})")
    
    def detect(self, 
               image_path: Path, 
               confidence_threshold: float = 0.3,
               save_visualization: bool = False) -> RTMDetResult:
        """
        检测图像中的物体
        
        Args:
            image_path: 图像路径
            confidence_threshold: 置信度阈值
            save_visualization: 是否保存可视化结果
            
        Returns:
            RTMDetResult: 检测结果
        """
        if self.model is None:
            # 如果模型未加载，返回模拟结果（用于演示）
            return self._mock_detect(image_path, confidence_threshold)
        
        import time
        start_time = time.time()
        
        # 推理
        result = self.inference_detector(self.model, str(image_path))
        
        # 解析结果
        detections = self._parse_results(result, confidence_threshold)
        
        # 可视化（可选）
        if save_visualization:
            self._save_visualization(image_path, result, confidence_threshold)
        
        inference_time = time.time() - start_time
        
        return RTMDetResult(
            detections=detections,
            image_path=str(image_path),
            inference_time=inference_time,
            model_info={
                'model': 'RTMDet-L',
                'mAP': 52.8,
                'license': 'Apache-2.0',
                'framework': 'MMDetection'
            }
        )
    
    def _parse_results(self, result, confidence_threshold: float) -> List[Dict]:
        """解析MMDetection推理结果"""
        detections = []
        
        # MMDetection 3.x版本的结果格式
        pred = result.pred_instances
        
        for bbox, score, label_id in zip(
            pred.bboxes.cpu().numpy(),
            pred.scores.cpu().numpy(), 
            pred.labels.cpu().numpy()
        ):
            if score < confidence_threshold:
                continue
            
            x1, y1, x2, y2 = bbox
            detection = {
                'class': self.COCO_CLASSES[label_id],
                'confidence': float(score),
                'bbox': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                },
                'category': self._get_category(self.COCO_CLASSES[label_id])
            }
            detections.append(detection)
        
        return detections
    
    def _get_category(self, class_name: str) -> str:
        """获取类别所属的大类"""
        for category, classes in self.PRIORITY_CLASSES.items():
            if class_name in classes:
                return category
        return '其他'
    
    def _save_visualization(self, image_path: Path, result, confidence_threshold: float):
        """保存检测结果的可视化图像"""
        output_path = image_path.parent / f"detected_{image_path.name}"
        self.model.show_result(
            str(image_path), 
            result, 
            score_thr=confidence_threshold,
            out_file=str(output_path)
        )
        print(f"可视化结果保存至: {output_path}")
    
    def _mock_detect(self, image_path: Path, confidence_threshold: float) -> RTMDetResult:
        """模拟检测（当MMDetection未安装时使用）"""
        import random
        import time
        
        # 模拟一些检测结果
        mock_detections = []
        
        # 随机生成2-5个检测结果
        num_detections = random.randint(2, 5)
        
        for i in range(num_detections):
            class_name = random.choice(['laptop', 'cell phone', 'cup', 'book', 'chair'])
            confidence = random.uniform(confidence_threshold, 0.95)
            
            mock_detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': {
                    'x1': random.uniform(0, 300),
                    'y1': random.uniform(0, 300),
                    'x2': random.uniform(400, 640),
                    'y2': random.uniform(400, 480)
                },
                'category': self._get_category(class_name)
            })
        
        return RTMDetResult(
            detections=mock_detections,
            image_path=str(image_path),
            inference_time=0.15,  # 模拟推理时间
            model_info={
                'model': 'RTMDet-L (模拟模式)',
                'mAP': 52.8,
                'license': 'Apache-2.0',
                'framework': 'MMDetection'
            }
        )
    
    def batch_detect(self, 
                    image_paths: List[Path],
                    confidence_threshold: float = 0.3,
                    batch_size: int = 4) -> List[RTMDetResult]:
        """
        批量检测多张图片
        
        Args:
            image_paths: 图像路径列表
            confidence_threshold: 置信度阈值
            batch_size: 批处理大小
            
        Returns:
            检测结果列表
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            
            for image_path in batch:
                result = self.detect(image_path, confidence_threshold)
                results.append(result)
        
        return results
    
    def analyze_for_media(self, image_path: Path) -> Dict:
        """
        为自媒体创作者分析图片
        提供更详细的内容分析和标签建议
        """
        result = self.detect(image_path, confidence_threshold=0.3)
        
        # 统计各类别物体
        category_counts = {}
        for detection in result.detections:
            category = detection['category']
            if category not in category_counts:
                category_counts[category] = []
            category_counts[category].append(detection['class'])
        
        # 生成标签建议
        tags = []
        for category, items in category_counts.items():
            tags.append(f"#{category}")
            for item in set(items):
                tags.append(f"#{item.replace(' ', '_')}")
        
        # 生成内容描述
        main_objects = [d['class'] for d in result.detections if d['confidence'] > 0.5]
        description = f"图片中包含: {', '.join(set(main_objects))}" if main_objects else "未检测到明显物体"
        
        return {
            'image_path': str(image_path),
            'category_summary': category_counts,
            'suggested_tags': tags,
            'description': description,
            'high_confidence_objects': [
                d for d in result.detections if d['confidence'] > 0.7
            ],
            'all_detections': result.detections,
            'model_info': result.model_info
        }


def demo():
    """演示函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python rtmdet_detector.py <image_path>")
        print("\n演示模式（无需MMDetection）:")
        print("将使用模拟数据展示功能")
        image_path = Path("demo.jpg")  # 虚拟路径
    else:
        image_path = Path(sys.argv[1])
        if not image_path.exists():
            print(f"图像未找到: {image_path}")
            return
    
    # 初始化检测器
    print("初始化RTMDet-L检测器...")
    detector = RTMDetDetector(device='cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 基础检测
    print("\n=== 基础物体检测 ===")
    result = detector.detect(image_path, confidence_threshold=0.3)
    
    print(f"检测到 {len(result.detections)} 个物体")
    print(f"推理时间: {result.inference_time:.3f}秒")
    
    for detection in result.detections[:5]:  # 显示前5个结果
        print(f"  - {detection['class']}: {detection['confidence']:.1%} "
              f"[类别: {detection['category']}]")
    
    # 自媒体分析
    print("\n=== 自媒体内容分析 ===")
    media_analysis = detector.analyze_for_media(image_path)
    
    print(f"内容描述: {media_analysis['description']}")
    print(f"建议标签: {' '.join(media_analysis['suggested_tags'][:10])}")
    
    if media_analysis['category_summary']:
        print("\n类别统计:")
        for category, items in media_analysis['category_summary'].items():
            print(f"  {category}: {', '.join(set(items))}")
    
    # 保存结果
    output_file = image_path.stem + '_rtmdet_detection.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'basic_detection': {
                'detections': result.detections,
                'inference_time': result.inference_time
            },
            'media_analysis': media_analysis,
            'model_info': result.model_info
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存至: {output_file}")
    
    # 对比说明
    print("\n=== RTMDet-L vs YOLO对比 ===")
    print("RTMDet-L优势:")
    print("  ✅ Apache-2.0许可 - 商用无忧")
    print("  ✅ 52.8% mAP - 更高精度")
    print("  ✅ OpenMMLab支持 - 社区活跃")
    print("  ✅ 80类COCO物体 - 覆盖广泛")
    print("\nYOLO限制:")
    print("  ❌ AGPL-3.0许可 - 商用需付费")
    print("  ❌ 50.2% mAP - 精度稍低")
    print("  ❌ 速度优先 - 牺牲精度")


if __name__ == "__main__":
    # 检查是否安装了必要的库
    try:
        import torch
    except ImportError:
        print("请先安装PyTorch: pip install torch")
        exit(1)
    
    demo()
