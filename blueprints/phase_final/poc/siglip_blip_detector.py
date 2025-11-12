"""
SigLIP + BLIP 检测器 POC - 展示多语言分类和图像理解功能

这个POC展示了如何使用SigLIP进行多语言图像分类，以及BLIP进行图像描述生成。
相比RTMDet，这个方案：
1. 无复杂依赖问题（mmcv已无法在新Python版本上安装）
2. 支持多语言（中文、英文、日文等）
3. 提供更丰富的语义理解
4. 可以生成自然语言描述
"""

import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from dataclasses import dataclass
from transformers import (
    AutoProcessor, 
    AutoModel,
    BlipProcessor, 
    BlipForConditionalGeneration
)
import numpy as np


@dataclass
class DetectionResult:
    """检测结果数据类"""
    image_path: str
    siglip_scores: Dict[str, float]  # 分类结果和置信度
    blip_caption: str  # BLIP生成的图像描述
    detected_objects: List[str]  # 检测到的物体
    confidence: float  # 总体置信度
    metadata: Dict[str, Any]  # 其他元数据


class SigLIPBLIPDetector:
    """
    基于SigLIP和BLIP的智能图像分析器
    
    特点：
    - 多语言支持（SigLIP i18n模型）
    - 图像理解和描述生成（BLIP）
    - 零样本分类能力
    - 无需复杂依赖（mmcv等）
    """
    
    def __init__(self, 
                 device: str = 'cpu',
                 siglip_model: str = "google/siglip-base-patch16-224-i18n",
                 blip_model: str = "Salesforce/blip-image-captioning-base"):
        """
        初始化检测器
        
        Args:
            device: 运行设备 ('cpu', 'cuda', 'mps')
            siglip_model: SigLIP模型名称
            blip_model: BLIP模型名称
        """
        self.device = device
        
        # 加载SigLIP模型（多语言分类）
        print(f"加载SigLIP模型: {siglip_model}")
        self.siglip_processor = AutoProcessor.from_pretrained(siglip_model)
        self.siglip_model = AutoModel.from_pretrained(siglip_model).to(device)
        
        # 加载BLIP模型（图像描述生成）
        print(f"加载BLIP模型: {blip_model}")
        self.blip_processor = BlipProcessor.from_pretrained(blip_model)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_model).to(device)
        
        print(f"模型加载成功 (设备: {self.device})")
        
    def detect(self, 
               image_path: Path,
               candidate_labels: List[str] = None,
               confidence_threshold: float = 0.3,
               generate_caption: bool = True) -> DetectionResult:
        """
        对单张图片进行检测和分析
        
        Args:
            image_path: 图片路径
            candidate_labels: 候选标签列表（支持多语言）
            confidence_threshold: 置信度阈值
            generate_caption: 是否生成图像描述
            
        Returns:
            DetectionResult: 检测结果
        """
        # 加载图片
        image = Image.open(image_path).convert('RGB')
        
        # 默认候选标签（支持多语言）
        if candidate_labels is None:
            candidate_labels = [
                # 电子产品（中英混合）
                "手机", "phone", "iPhone", "安卓手机", "Android phone",
                "电脑", "computer", "laptop", "MacBook", "笔记本电脑",
                "平板", "tablet", "iPad",
                "耳机", "headphones", "AirPods",
                "相机", "camera",
                
                # 美食（中英混合）
                "美食", "food", "食物",
                "披萨", "pizza",
                "汉堡", "burger", "hamburger",
                "寿司", "sushi",
                "面条", "noodles", "拉面", "pasta",
                "蛋糕", "cake", "甜点", "dessert",
                
                # 文档
                "文档", "document", "文件", "paper",
                "书", "book", "书籍",
                "笔记", "notes",
                
                # 人物和场景
                "人", "person", "people",
                "风景", "landscape", "scenery",
                "建筑", "building", "architecture",
                "动物", "animal", "pet", "宠物"
            ]
        
        # 使用SigLIP进行零样本分类
        siglip_results = self._classify_with_siglip(image, candidate_labels)
        
        # 使用BLIP生成图像描述
        caption = ""
        if generate_caption:
            caption = self._generate_caption_with_blip(image)
        
        # 筛选高置信度的检测结果
        detected_objects = [
            label for label, score in siglip_results.items() 
            if score >= confidence_threshold
        ]
        
        # 计算总体置信度
        top_scores = sorted(siglip_results.values(), reverse=True)[:3]
        overall_confidence = np.mean(top_scores) if top_scores else 0.0
        
        return DetectionResult(
            image_path=str(image_path),
            siglip_scores=siglip_results,
            blip_caption=caption,
            detected_objects=detected_objects,
            confidence=float(overall_confidence),
            metadata={
                'model': 'SigLIP+BLIP',
                'siglip_model': self.siglip_processor.name_or_path,
                'blip_model': self.blip_processor.name_or_path,
                'device': self.device
            }
        )
    
    def _classify_with_siglip(self, image: Image, labels: List[str]) -> Dict[str, float]:
        """使用SigLIP进行零样本分类"""
        inputs = self.siglip_processor(
            text=labels,
            images=image,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.siglip_model(**inputs)
            # 获取图像-文本相似度
            logits_per_image = outputs.logits_per_image
            probs = torch.sigmoid(logits_per_image)  # SigLIP使用sigmoid而不是softmax
        
        # 转换为字典格式
        results = {}
        for label, prob in zip(labels, probs[0].cpu().numpy()):
            results[label] = float(prob)
        
        return results
    
    def _generate_caption_with_blip(self, image: Image) -> str:
        """使用BLIP生成图像描述"""
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # 生成描述
            out = self.blip_model.generate(**inputs, max_length=50)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        
        return caption
    
    def batch_detect(self, 
                    image_paths: List[Path],
                    batch_size: int = 4,
                    **kwargs) -> List[DetectionResult]:
        """
        批量检测图片
        
        Args:
            image_paths: 图片路径列表
            batch_size: 批处理大小
            **kwargs: 传递给detect方法的其他参数
            
        Returns:
            List[DetectionResult]: 检测结果列表
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            for path in batch:
                try:
                    result = self.detect(path, **kwargs)
                    results.append(result)
                    print(f"处理完成: {path.name}")
                except Exception as e:
                    print(f"处理失败 {path}: {e}")
                    
        return results
    
    def analyze_for_specific_product(self, 
                                    image_path: Path,
                                    product_keywords: List[str]) -> Tuple[bool, float, str]:
        """
        检查图片是否包含特定产品
        
        Args:
            image_path: 图片路径
            product_keywords: 产品关键词列表
            
        Returns:
            Tuple[是否包含, 置信度, 描述]
        """
        result = self.detect(image_path, candidate_labels=product_keywords)
        
        # 检查是否有高置信度的匹配
        max_score = max(result.siglip_scores.values()) if result.siglip_scores else 0
        contains_product = max_score > 0.5
        
        # 生成描述
        if contains_product:
            top_match = max(result.siglip_scores.items(), key=lambda x: x[1])
            description = f"检测到: {top_match[0]} (置信度: {top_match[1]:.2%})"
            if result.blip_caption:
                description += f"\n描述: {result.blip_caption}"
        else:
            description = "未检测到指定产品"
            
        return contains_product, max_score, description


def demo():
    """演示SigLIP+BLIP检测器的功能"""
    
    print("=== SigLIP+BLIP 检测器演示 ===\n")
    
    # 初始化检测器
    print("初始化检测器...")
    detector = SigLIPBLIPDetector(
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    
    # 测试图片路径（请替换为实际路径）
    test_images = [
        Path("./sample_data/iphone.jpg"),
        Path("./sample_data/pizza.jpg"),
        Path("./sample_data/document.pdf"),
    ]
    
    # 模拟测试（当文件不存在时）
    if not any(p.exists() for p in test_images):
        print("\n未找到测试图片，使用模拟数据演示...\n")
        
        # 模拟结果展示
        mock_results = [
            {
                'image': 'iphone.jpg',
                'top_detections': {'iPhone': 0.92, '手机': 0.89, 'phone': 0.88},
                'caption': 'a close up of a cell phone on a table',
                'confidence': 0.90
            },
            {
                'image': 'pizza.jpg', 
                'top_detections': {'披萨': 0.95, 'pizza': 0.94, '美食': 0.87},
                'caption': 'a pizza with cheese and vegetables on it',
                'confidence': 0.92
            },
            {
                'image': 'document.pdf',
                'top_detections': {'文档': 0.85, 'document': 0.83, '文件': 0.80},
                'caption': 'a piece of paper with text on it',
                'confidence': 0.83
            }
        ]
        
        for result in mock_results:
            print(f"图片: {result['image']}")
            print(f"主要检测结果:")
            for label, score in result['top_detections'].items():
                print(f"  - {label}: {score:.2%}")
            print(f"图像描述: {result['caption']}")
            print(f"总体置信度: {result['confidence']:.2%}")
            print("-" * 50)
    
    else:
        # 实际检测
        print("\n开始检测...\n")
        results = detector.batch_detect(test_images)
        
        # 展示结果
        for result in results:
            print(f"图片: {result.image_path}")
            print(f"检测到的物体: {', '.join(result.detected_objects)}")
            print(f"图像描述: {result.blip_caption}")
            print(f"总体置信度: {result.confidence:.2%}")
            
            # 展示top-5分类结果
            top_5 = sorted(result.siglip_scores.items(), 
                         key=lambda x: x[1], reverse=True)[:5]
            print("Top-5 分类结果:")
            for label, score in top_5:
                print(f"  - {label}: {score:.2%}")
            print("-" * 50)
    
    # 特定产品检测示例
    print("\n=== 特定产品检测示例 ===")
    print("搜索关键词: ['iPhone 15', 'iPhone 14', 'iPhone', '苹果手机']")
    print("模拟结果:")
    print("  ✓ 找到匹配: iPhone (置信度: 92%)")
    print("  描述: a close up of an iPhone on a wooden table")
    
    print("\n=== SigLIP+BLIP vs RTMDet 对比 ===")
    print("SigLIP+BLIP优势:")
    print("  1. ✅ 无复杂依赖（mmcv无法在Python 3.11+安装）")
    print("  2. ✅ 多语言支持（中文、英文、日文等）")
    print("  3. ✅ 零样本学习（无需预定义类别）") 
    print("  4. ✅ 自然语言描述生成")
    print("  5. ✅ 更好的语义理解")
    print("\nRTMDet劣势:")
    print("  1. ❌ mmcv依赖问题严重")
    print("  2. ❌ 仅支持预定义的80个COCO类别")
    print("  3. ❌ 无法生成描述")
    print("  4. ❌ 不支持多语言")


if __name__ == "__main__":
    demo()
