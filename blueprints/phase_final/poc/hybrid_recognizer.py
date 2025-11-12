#!/usr/bin/env python3
"""
混合识别器 POC - 展示AI+人工的协同工作模式
演示如何平衡自动化和人工介入
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from datetime import datetime


class ConfidenceLevel(Enum):
    """置信度级别"""
    HIGH = "high"        # > 0.8  - 自动处理
    MEDIUM = "medium"    # 0.5-0.8 - AI建议
    LOW = "low"          # < 0.5  - 需要人工


@dataclass
class RecognitionResult:
    """识别结果"""
    # 基础信息
    image_path: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # AI预测
    ai_category: Optional[str] = None
    ai_confidence: Optional[float] = None
    ai_suggestions: List[Dict] = field(default_factory=list)
    
    # 人工标注
    human_label: Optional[str] = None
    human_verified: bool = False
    
    # 决策信息
    confidence_level: Optional[ConfidenceLevel] = None
    action_taken: Optional[str] = None
    needs_review: bool = True
    
    # 学习相关
    used_for_training: bool = False
    similar_images: List[str] = field(default_factory=list)


class HybridRecognizer:
    """
    混合识别器：AI + 人工的最佳配合
    """
    
    def __init__(self):
        """初始化识别器"""
        # 模拟的识别阈值
        self.thresholds = {
            'auto_accept': 0.8,     # 自动接受
            'suggest': 0.5,         # 提供建议
            'reject': 0.3           # 拒绝，需要人工
        }
        
        # 模拟的已学习模式
        self.learned_patterns = {}
        
        # 标注历史（用于学习）
        self.annotation_history = []
        
        # 用户偏好
        self.user_preferences = {
            'common_labels': ['iPhone', 'MacBook', '披萨', '截图'],
            'recent_labels': [],
            'label_shortcuts': {
                '1': 'iPhone',
                '2': 'MacBook', 
                '3': '披萨',
                '4': '文档'
            }
        }
    
    def recognize(self, image_path: str, 
                 ai_prediction: Tuple[str, float]) -> RecognitionResult:
        """
        执行混合识别
        
        Args:
            image_path: 图像路径
            ai_prediction: AI预测结果 (类别, 置信度)
            
        Returns:
            识别结果
        """
        category, confidence = ai_prediction
        
        # 创建结果对象
        result = RecognitionResult(
            image_path=image_path,
            ai_category=category,
            ai_confidence=confidence
        )
        
        # 判断置信度级别
        result.confidence_level = self._get_confidence_level(confidence)
        
        # 根据置信度决定行动
        if result.confidence_level == ConfidenceLevel.HIGH:
            # 高置信度：自动接受
            result.action_taken = "auto_accepted"
            result.human_label = category
            result.needs_review = False
            result.human_verified = True
            print(f"✅ 自动接受: {category} ({confidence:.1%})")
            
        elif result.confidence_level == ConfidenceLevel.MEDIUM:
            # 中置信度：提供建议
            result.action_taken = "suggested"
            result.ai_suggestions = self._generate_suggestions(image_path, category)
            result.needs_review = True
            print(f"💡 AI建议: {category} ({confidence:.1%})")
            print(f"   其他可能: {result.ai_suggestions}")
            
        else:
            # 低置信度：需要人工
            result.action_taken = "manual_required"
            result.needs_review = True
            print(f"❓ 需要人工标注 (AI猜测: {category} - {confidence:.1%})")
        
        # 查找相似图片
        result.similar_images = self._find_similar_images(image_path)
        if result.similar_images:
            print(f"📷 找到 {len(result.similar_images)} 张相似图片")
        
        return result
    
    def apply_human_annotation(self, result: RecognitionResult, 
                              human_label: str) -> RecognitionResult:
        """
        应用人工标注
        
        Args:
            result: 原始识别结果
            human_label: 人工标注
            
        Returns:
            更新后的结果
        """
        result.human_label = human_label
        result.human_verified = True
        result.needs_review = False
        
        # 记录标注历史
        self.annotation_history.append({
            'image': result.image_path,
            'ai_prediction': result.ai_category,
            'ai_confidence': result.ai_confidence,
            'human_label': human_label,
            'timestamp': result.timestamp
        })
        
        # 更新用户偏好
        if human_label not in self.user_preferences['recent_labels']:
            self.user_preferences['recent_labels'].insert(0, human_label)
            self.user_preferences['recent_labels'] = \
                self.user_preferences['recent_labels'][:10]  # 保留最近10个
        
        # 如果人工标注与AI不同，标记用于学习
        if human_label != result.ai_category:
            result.used_for_training = True
            self._learn_from_correction(result)
        
        print(f"✏️ 人工标注: {human_label}")
        
        return result
    
    def batch_apply(self, primary_result: RecognitionResult, 
                   similar_images: List[str]) -> List[RecognitionResult]:
        """
        批量应用标注到相似图片
        
        Args:
            primary_result: 主图片的结果
            similar_images: 相似图片列表
            
        Returns:
            批量处理结果
        """
        if not primary_result.human_verified:
            print("⚠️ 主图片未经人工确认，无法批量应用")
            return []
        
        results = []
        for img_path in similar_images:
            # 创建新结果
            batch_result = RecognitionResult(
                image_path=img_path,
                ai_category=primary_result.ai_category,
                ai_confidence=0.95,  # 基于相似性的高置信度
                human_label=primary_result.human_label,
                human_verified=True,
                needs_review=False,
                action_taken="batch_applied"
            )
            results.append(batch_result)
        
        print(f"🎯 批量应用标签 '{primary_result.human_label}' 到 {len(results)} 张图片")
        
        return results
    
    def generate_annotation_ui(self, result: RecognitionResult) -> Dict:
        """
        生成标注界面数据
        
        模拟一个智能的标注界面
        """
        ui_data = {
            'image': result.image_path,
            'ai_prediction': {
                'category': result.ai_category,
                'confidence': result.ai_confidence,
                'level': result.confidence_level.value if result.confidence_level else None
            },
            'suggestions': [],
            'shortcuts': self.user_preferences['label_shortcuts'],
            'recent_labels': self.user_preferences['recent_labels'],
            'similar_count': len(result.similar_images),
            'actions': []
        }
        
        # 根据置信度生成不同的UI
        if result.confidence_level == ConfidenceLevel.HIGH:
            ui_data['actions'] = [
                {'key': 'Space', 'action': '确认', 'primary': True},
                {'key': 'X', 'action': '跳过'},
                {'key': 'E', 'action': '编辑'}
            ]
            ui_data['message'] = f"AI高度确信这是: {result.ai_category}"
            
        elif result.confidence_level == ConfidenceLevel.MEDIUM:
            # 提供多个选项
            ui_data['suggestions'] = [
                result.ai_category,
                *[s['label'] for s in result.ai_suggestions[:3]]
            ]
            ui_data['actions'] = [
                {'key': '1-4', 'action': '选择建议'},
                {'key': 'T', 'action': '输入自定义'},
                {'key': 'X', 'action': '跳过'}
            ]
            ui_data['message'] = "AI不太确定，请选择或输入"
            
        else:
            ui_data['actions'] = [
                {'key': '1-9', 'action': '快捷标签'},
                {'key': 'T', 'action': '输入标签'},
                {'key': 'X', 'action': '标记为未知'}
            ]
            ui_data['message'] = "AI无法识别，请手动标注"
        
        # 添加批量操作选项
        if result.similar_images:
            ui_data['batch_option'] = {
                'available': True,
                'count': len(result.similar_images),
                'key': 'G',
                'action': '应用到所有相似图片'
            }
        
        return ui_data
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """判断置信度级别"""
        if confidence >= self.thresholds['auto_accept']:
            return ConfidenceLevel.HIGH
        elif confidence >= self.thresholds['suggest']:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _generate_suggestions(self, image_path: str, 
                            primary_category: str) -> List[Dict]:
        """生成备选建议"""
        # 模拟生成建议
        suggestions = []
        
        # 基于历史的建议
        if self.user_preferences['recent_labels']:
            suggestions.append({
                'label': self.user_preferences['recent_labels'][0],
                'reason': 'recently_used',
                'score': 0.7
            })
        
        # 基于相似性的建议
        similar_categories = {
            '手机': ['iPhone', 'Samsung', 'Android手机'],
            '电脑': ['MacBook', 'ThinkPad', '笔记本'],
            '美食': ['披萨', '汉堡', '面条']
        }
        
        if primary_category in similar_categories:
            for cat in similar_categories[primary_category][:2]:
                suggestions.append({
                    'label': cat,
                    'reason': 'similar_category',
                    'score': 0.6
                })
        
        return suggestions
    
    def _find_similar_images(self, image_path: str) -> List[str]:
        """查找相似图片"""
        # 模拟查找相似图片
        # 实际实现中会使用向量相似度
        import random
        
        if random.random() > 0.5:
            count = random.randint(1, 10)
            return [f"similar_{i}.jpg" for i in range(count)]
        return []
    
    def _learn_from_correction(self, result: RecognitionResult):
        """从纠正中学习"""
        # 模拟学习过程
        key = f"{result.ai_category}->{result.human_label}"
        
        if key not in self.learned_patterns:
            self.learned_patterns[key] = 0
        self.learned_patterns[key] += 1
        
        print(f"🧠 学习模式: {key} (已见{self.learned_patterns[key]}次)")
        
        # 如果某个模式出现多次，可以调整阈值或添加规则
        if self.learned_patterns[key] >= 5:
            print(f"💡 检测到频繁纠正模式，建议重新训练模型")
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total = len(self.annotation_history)
        
        if total == 0:
            return {'message': '暂无标注历史'}
        
        correct = sum(1 for a in self.annotation_history 
                     if a['ai_prediction'] == a['human_label'])
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'total_annotations': total,
            'ai_correct': correct,
            'ai_accuracy': accuracy,
            'learned_patterns': len(self.learned_patterns),
            'common_corrections': sorted(
                self.learned_patterns.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }


def demo():
    """演示混合识别流程"""
    
    print("=== 混合识别器演示 ===\n")
    
    recognizer = HybridRecognizer()
    
    # 模拟不同置信度的场景
    test_cases = [
        ("photo1.jpg", ("iPhone", 0.92)),      # 高置信度
        ("photo2.jpg", ("手机", 0.65)),        # 中置信度
        ("photo3.jpg", ("未知物体", 0.25)),    # 低置信度
        ("photo4.jpg", ("MacBook", 0.88)),     # 高置信度
        ("photo5.jpg", ("电脑", 0.55)),        # 中置信度
    ]
    
    results = []
    
    for image_path, ai_prediction in test_cases:
        print(f"\n--- 处理: {image_path} ---")
        
        # 执行识别
        result = recognizer.recognize(image_path, ai_prediction)
        
        # 如果需要人工介入，模拟标注
        if result.needs_review:
            # 生成UI数据
            ui_data = recognizer.generate_annotation_ui(result)
            print(f"UI提示: {ui_data['message']}")
            
            # 模拟人工标注
            if result.confidence_level == ConfidenceLevel.MEDIUM:
                # 中置信度，确认AI的建议
                human_label = result.ai_category
            else:
                # 低置信度，提供新标签
                human_label = "专业设备"
            
            result = recognizer.apply_human_annotation(result, human_label)
            
            # 如果有相似图片，批量应用
            if result.similar_images:
                batch_results = recognizer.batch_apply(result, result.similar_images)
                results.extend(batch_results)
        
        results.append(result)
    
    # 显示统计
    print("\n=== 统计信息 ===")
    stats = recognizer.get_statistics()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # 保存结果
    output = {
        'results': [
            {
                'image': r.image_path,
                'ai_category': r.ai_category,
                'ai_confidence': r.ai_confidence,
                'human_label': r.human_label,
                'action': r.action_taken,
                'verified': r.human_verified
            }
            for r in results
        ],
        'statistics': stats
    }
    
    with open('hybrid_recognition_results.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("\n✅ 结果已保存到: hybrid_recognition_results.json")


if __name__ == "__main__":
    demo()
