#!/usr/bin/env python3
"""
Few-Shot学习器 POC - 展示如何用少量样本学习新产品
5-10个样本就能识别用户的专业设备
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from datetime import datetime


@dataclass
class ProductPrototype:
    """产品原型 - 代表一个已学习的产品"""
    name: str
    category: str
    feature_vector: np.ndarray
    sample_count: int
    threshold: float
    created_at: str
    accuracy: Optional[float] = None
    
    def to_dict(self):
        """转换为可序列化的字典"""
        return {
            'name': self.name,
            'category': self.category,
            'sample_count': self.sample_count,
            'threshold': self.threshold,
            'created_at': self.created_at,
            'accuracy': self.accuracy
        }


class FewShotLearner:
    """
    Few-Shot学习器
    使用少量样本学习识别新产品
    """
    
    def __init__(self, feature_dim: int = 512):
        """
        初始化学习器
        
        Args:
            feature_dim: 特征向量维度
        """
        self.feature_dim = feature_dim
        self.prototypes = {}  # 存储已学习的产品原型
        self.min_samples = 3  # 最少样本数
        self.max_samples = 20  # 最多样本数
        
        # 模拟的特征提取器（实际使用DINOv2或SigLIP）
        self.feature_extractor = self._mock_feature_extractor
        
        print(f"Few-Shot学习器初始化完成 (特征维度: {feature_dim})")
    
    def learn_new_product(self, 
                         product_name: str,
                         category: str,
                         sample_images: List[str]) -> ProductPrototype:
        """
        学习新产品
        
        Args:
            product_name: 产品名称，如 "iPhone 15 Pro"
            category: 产品类别，如 "手机"
            sample_images: 样本图片路径列表
            
        Returns:
            学习后的产品原型
        """
        # 检查样本数量
        n_samples = len(sample_images)
        if n_samples < self.min_samples:
            raise ValueError(f"至少需要{self.min_samples}个样本，当前只有{n_samples}个")
        
        if n_samples > self.max_samples:
            print(f"样本数超过{self.max_samples}，只使用前{self.max_samples}个")
            sample_images = sample_images[:self.max_samples]
        
        print(f"\n开始学习新产品: {product_name}")
        print(f"类别: {category}")
        print(f"样本数: {len(sample_images)}")
        
        # Step 1: 提取特征
        features = []
        for img_path in sample_images:
            feature = self.feature_extractor(img_path)
            features.append(feature)
            print(f"  ✓ 提取特征: {img_path}")
        
        features = np.array(features)
        
        # Step 2: 计算原型（特征的平均值）
        prototype_vector = np.mean(features, axis=0)
        print(f"原型向量计算完成 (维度: {prototype_vector.shape})")
        
        # Step 3: 计算阈值（基于样本间的距离）
        threshold = self._calculate_threshold(features, prototype_vector)
        print(f"识别阈值: {threshold:.4f}")
        
        # Step 4: 创建原型
        prototype = ProductPrototype(
            name=product_name,
            category=category,
            feature_vector=prototype_vector,
            sample_count=len(sample_images),
            threshold=threshold,
            created_at=datetime.now().isoformat()
        )
        
        # Step 5: 验证学习效果
        accuracy = self._validate_prototype(prototype, features)
        prototype.accuracy = accuracy
        print(f"自验证准确率: {accuracy:.1%}")
        
        # Step 6: 保存原型
        self.prototypes[product_name] = prototype
        print(f"✅ 成功学习产品: {product_name}")
        
        return prototype
    
    def recognize(self, image_path: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        识别图片中的产品
        
        Args:
            image_path: 待识别图片路径
            top_k: 返回前K个可能的产品
            
        Returns:
            产品名称和相似度列表
        """
        if not self.prototypes:
            return [("未学习任何产品", 0.0)]
        
        # 提取特征
        feature = self.feature_extractor(image_path)
        
        # 计算与所有原型的相似度
        similarities = []
        for product_name, prototype in self.prototypes.items():
            similarity = self._compute_similarity(feature, prototype.feature_vector)
            
            # 检查是否超过阈值
            if similarity >= prototype.threshold:
                similarities.append((product_name, similarity))
        
        # 排序并返回
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        if not similarities:
            return [("未识别到已学习的产品", 0.0)]
        
        return similarities[:top_k]
    
    def update_product(self, product_name: str, new_samples: List[str]):
        """
        用新样本更新已学习的产品
        
        增量学习，不需要重新学习所有样本
        """
        if product_name not in self.prototypes:
            raise ValueError(f"产品 {product_name} 尚未学习")
        
        prototype = self.prototypes[product_name]
        print(f"\n更新产品: {product_name}")
        print(f"当前样本数: {prototype.sample_count}")
        print(f"新增样本数: {len(new_samples)}")
        
        # 提取新样本的特征
        new_features = []
        for img_path in new_samples:
            feature = self.feature_extractor(img_path)
            new_features.append(feature)
        new_features = np.array(new_features)
        
        # 增量更新原型（加权平均）
        old_weight = prototype.sample_count
        new_weight = len(new_samples)
        total_weight = old_weight + new_weight
        
        updated_prototype = (
            prototype.feature_vector * old_weight + 
            np.mean(new_features, axis=0) * new_weight
        ) / total_weight
        
        # 更新原型
        prototype.feature_vector = updated_prototype
        prototype.sample_count = total_weight
        
        # 重新计算阈值
        all_features = np.vstack([
            np.tile(prototype.feature_vector, (old_weight, 1)),  # 模拟旧样本
            new_features
        ])
        prototype.threshold = self._calculate_threshold(all_features, updated_prototype)
        
        print(f"✅ 产品更新完成，总样本数: {prototype.sample_count}")
    
    def remove_product(self, product_name: str):
        """删除已学习的产品"""
        if product_name in self.prototypes:
            del self.prototypes[product_name]
            print(f"已删除产品: {product_name}")
        else:
            print(f"产品不存在: {product_name}")
    
    def list_products(self) -> List[Dict]:
        """列出所有已学习的产品"""
        products = []
        for name, prototype in self.prototypes.items():
            products.append({
                'name': name,
                'category': prototype.category,
                'samples': prototype.sample_count,
                'accuracy': prototype.accuracy,
                'created': prototype.created_at
            })
        return products
    
    def save_model(self, path: str):
        """保存学习的模型"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(self.prototypes, f)
        print(f"模型已保存到: {save_path}")
    
    def load_model(self, path: str):
        """加载已保存的模型"""
        with open(path, 'rb') as f:
            self.prototypes = pickle.load(f)
        print(f"模型已加载，包含 {len(self.prototypes)} 个产品")
    
    def _mock_feature_extractor(self, image_path: str) -> np.ndarray:
        """
        模拟的特征提取器
        实际应使用 DINOv2, SigLIP 等预训练模型
        """
        # 模拟：基于文件名生成特征
        np.random.seed(hash(image_path) % 2**32)
        
        # 生成基础特征
        base_feature = np.random.randn(self.feature_dim)
        
        # 根据文件名添加一些模式
        if 'iphone' in image_path.lower():
            base_feature[:10] += 2.0  # iPhone 特征
        elif 'macbook' in image_path.lower():
            base_feature[10:20] += 2.0  # MacBook 特征
        elif 'pizza' in image_path.lower():
            base_feature[20:30] += 2.0  # 披萨特征
        
        # 归一化
        feature = base_feature / np.linalg.norm(base_feature)
        
        return feature
    
    def _compute_similarity(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """计算两个特征向量的相似度（余弦相似度）"""
        similarity = np.dot(feature1, feature2)
        # 转换到 [0, 1] 范围
        return (similarity + 1.0) / 2.0
    
    def _calculate_threshold(self, features: np.ndarray, prototype: np.ndarray) -> float:
        """
        计算识别阈值
        基于样本到原型的距离分布
        """
        # 计算所有样本到原型的相似度
        similarities = []
        for feature in features:
            sim = self._compute_similarity(feature, prototype)
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # 使用均值减去标准差作为阈值
        # 这样可以容纳一定的变化
        threshold = np.mean(similarities) - np.std(similarities)
        
        # 确保阈值在合理范围内
        threshold = max(0.3, min(0.9, threshold))
        
        return threshold
    
    def _validate_prototype(self, prototype: ProductPrototype, 
                          features: np.ndarray) -> float:
        """验证原型的识别效果"""
        correct = 0
        for feature in features:
            similarity = self._compute_similarity(feature, prototype.feature_vector)
            if similarity >= prototype.threshold:
                correct += 1
        
        return correct / len(features)


def demo():
    """演示Few-Shot学习流程"""
    
    print("=== Few-Shot学习器演示 ===\n")
    
    # 创建学习器
    learner = FewShotLearner(feature_dim=128)
    
    # 场景1: 学习新的专业设备
    print("\n场景1: 学习专业示波器")
    oscilloscope_samples = [
        "keysight_dso_1.jpg",
        "keysight_dso_2.jpg",
        "keysight_dso_3.jpg",
        "keysight_dso_4.jpg",
        "keysight_dso_5.jpg",
    ]
    
    prototype1 = learner.learn_new_product(
        "Keysight DSO-X 3024T",
        "测试仪器",
        oscilloscope_samples
    )
    
    # 场景2: 学习罕见的机械键盘
    print("\n场景2: 学习定制机械键盘")
    keyboard_samples = [
        "custom_keyboard_1.jpg",
        "custom_keyboard_2.jpg",
        "custom_keyboard_3.jpg",
    ]
    
    prototype2 = learner.learn_new_product(
        "HHKB Professional Hybrid Type-S",
        "键盘",
        keyboard_samples
    )
    
    # 场景3: 学习特殊的披萨
    print("\n场景3: 学习那不勒斯披萨")
    pizza_samples = [
        "neapolitan_pizza_1.jpg",
        "neapolitan_pizza_2.jpg",
        "neapolitan_pizza_3.jpg",
        "neapolitan_pizza_4.jpg",
    ]
    
    prototype3 = learner.learn_new_product(
        "那不勒斯玛格丽特披萨",
        "美食",
        pizza_samples
    )
    
    # 测试识别
    print("\n=== 测试识别 ===")
    
    test_images = [
        "keysight_dso_test.jpg",  # 应该识别为示波器
        "random_keyboard.jpg",     # 可能识别为键盘
        "neapolitan_pizza_new.jpg", # 应该识别为那不勒斯披萨
        "unknown_device.jpg",      # 不应该识别
    ]
    
    for img in test_images:
        print(f"\n测试图片: {img}")
        results = learner.recognize(img, top_k=2)
        for product, similarity in results:
            print(f"  - {product}: {similarity:.1%}")
    
    # 增量更新
    print("\n=== 增量学习 ===")
    new_samples = ["keysight_dso_6.jpg", "keysight_dso_7.jpg"]
    learner.update_product("Keysight DSO-X 3024T", new_samples)
    
    # 显示所有已学习的产品
    print("\n=== 已学习的产品 ===")
    products = learner.list_products()
    for product in products:
        print(f"- {product['name']}")
        print(f"  类别: {product['category']}")
        print(f"  样本数: {product['samples']}")
        print(f"  准确率: {product['accuracy']:.1%}")
    
    # 保存模型
    learner.save_model("few_shot_model.pkl")
    
    # 导出统计
    stats = {
        'total_products': len(learner.prototypes),
        'products': products,
        'feature_dim': learner.feature_dim,
        'min_samples': learner.min_samples,
    }
    
    with open('few_shot_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print("\n✅ 演示完成！")
    print("- 模型已保存到: few_shot_model.pkl")
    print("- 统计已保存到: few_shot_stats.json")


if __name__ == "__main__":
    demo()
