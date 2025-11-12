#!/usr/bin/env python3
"""
简单检测器 POC - 展示基础分类功能
可以直接运行测试：python simple_detector.py /path/to/image.jpg
"""

from pathlib import Path
from typing import List, Dict, Optional
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from dataclasses import dataclass
import json


@dataclass
class DetectionResult:
    """检测结果数据类"""
    category: str
    confidence: float
    alternatives: List[Dict[str, float]]
    metadata: Dict


class SimpleDetector:
    """
    极简的图像检测器
    使用CLIP模型进行零样本分类
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """初始化检测器"""
        print(f"Loading model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # 预定义的类别体系
        self.categories = {
            'general': [
                '电子产品', '美食', '文档', '风景', 
                '人物', '动物', '建筑', '其他'
            ],
            'electronics': [
                '手机', '电脑', '平板', '相机', 
                '耳机', '手表', '显示器', '键盘'
            ],
            'food': [
                '披萨', '汉堡', '面条', '米饭', 
                '甜点', '水果', '饮料', '蔬菜'
            ],
            'specific_products': [
                'iPhone', 'MacBook', 'iPad', 'AirPods',
                'Samsung Galaxy', 'ThinkPad', 'Surface'
            ]
        }
        
    def detect(self, image_path: Path, 
               category_set: str = 'general',
               threshold: float = 0.3) -> DetectionResult:
        """
        检测图像类别
        
        Args:
            image_path: 图像路径
            category_set: 使用的类别集合
            threshold: 置信度阈值
            
        Returns:
            DetectionResult: 检测结果
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 获取类别列表
        categories = self.categories.get(category_set, self.categories['general'])
        
        # 准备输入
        inputs = self.processor(
            text=categories,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # 获取结果
        confidences = probs[0].tolist()
        results = list(zip(categories, confidences))
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 构建返回结果
        main_category = results[0][0]
        main_confidence = results[0][1]
        
        # 获取备选结果
        alternatives = [
            {'category': cat, 'confidence': conf}
            for cat, conf in results[1:4]  # Top 3 alternatives
        ]
        
        # 元数据
        metadata = {
            'image_size': image.size,
            'threshold_met': main_confidence >= threshold,
            'needs_review': main_confidence < 0.5,
            'model': 'clip-vit-base-patch32'
        }
        
        return DetectionResult(
            category=main_category,
            confidence=main_confidence,
            alternatives=alternatives,
            metadata=metadata
        )
    
    def detect_hierarchical(self, image_path: Path) -> Dict:
        """
        分层检测：从粗到细
        
        先检测大类，再检测子类
        """
        # Step 1: 检测大类
        general_result = self.detect(image_path, 'general')
        
        results = {
            'level1': {
                'category': general_result.category,
                'confidence': general_result.confidence
            }
        }
        
        # Step 2: 根据大类检测子类
        if general_result.category == '电子产品' and general_result.confidence > 0.5:
            electronic_result = self.detect(image_path, 'electronics')
            results['level2'] = {
                'category': electronic_result.category,
                'confidence': electronic_result.confidence
            }
            
            # Step 3: 尝试识别具体产品
            if electronic_result.category in ['手机', '电脑', '平板']:
                product_result = self.detect(image_path, 'specific_products')
                results['level3'] = {
                    'category': product_result.category,
                    'confidence': product_result.confidence
                }
        
        elif general_result.category == '美食' and general_result.confidence > 0.5:
            food_result = self.detect(image_path, 'food')
            results['level2'] = {
                'category': food_result.category,
                'confidence': food_result.confidence
            }
        
        return results
    
    def batch_detect(self, image_paths: List[Path], 
                    batch_size: int = 8) -> List[DetectionResult]:
        """
        批量检测多张图片
        
        Args:
            image_paths: 图像路径列表
            batch_size: 批次大小
            
        Returns:
            检测结果列表
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            
            # 加载批次图像
            images = [Image.open(p).convert('RGB') for p in batch]
            
            # 批量处理
            categories = self.categories['general']
            inputs = self.processor(
                text=categories,
                images=images,
                return_tensors="pt",
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits_per_image
                probs = logits.softmax(dim=1)
            
            # 解析每张图片的结果
            for j, path in enumerate(batch):
                confidences = probs[j].tolist()
                img_results = list(zip(categories, confidences))
                img_results.sort(key=lambda x: x[1], reverse=True)
                
                results.append(DetectionResult(
                    category=img_results[0][0],
                    confidence=img_results[0][1],
                    alternatives=[
                        {'category': cat, 'confidence': conf}
                        for cat, conf in img_results[1:4]
                    ],
                    metadata={'path': str(path)}
                ))
        
        return results


def demo():
    """演示函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python simple_detector.py <image_path>")
        return
    
    image_path = Path(sys.argv[1])
    
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return
    
    # 初始化检测器
    print("Initializing detector...")
    detector = SimpleDetector()
    
    # 简单检测
    print("\n=== Simple Detection ===")
    result = detector.detect(image_path)
    print(f"Category: {result.category} ({result.confidence:.1%})")
    print(f"Alternatives:")
    for alt in result.alternatives:
        print(f"  - {alt['category']}: {alt['confidence']:.1%}")
    
    # 分层检测
    print("\n=== Hierarchical Detection ===")
    hierarchical = detector.detect_hierarchical(image_path)
    for level, info in hierarchical.items():
        print(f"{level}: {info['category']} ({info['confidence']:.1%})")
    
    # 判断是否需要人工确认
    if result.metadata['needs_review']:
        print("\n⚠️ Low confidence - needs human review")
    else:
        print("\n✅ High confidence - auto-labeled")
    
    # 保存结果
    output = {
        'image': str(image_path),
        'detection': {
            'category': result.category,
            'confidence': result.confidence,
            'alternatives': result.alternatives
        },
        'hierarchical': hierarchical,
        'metadata': result.metadata
    }
    
    output_file = image_path.stem + '_detection.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    demo()
