# 视频处理策略与实施方案

## 📹 视频处理范围界定

### 当前数据集情况
- **视频文件数量**: 762个MOV文件
- **占总文件比例**: 2.5% (762/30,874)
- **预估大小**: 约50-100GB（基于MOV格式特性）
- **时间跨度**: 分布在14年的照片库中

### 处理策略决策

#### 方案A：MVP阶段排除视频（推荐）✅

**理由**：
1. 视频仅占2.5%，对MVP验证影响小
2. 集中资源优化图片处理（97.5%的内容）
3. 降低技术复杂度和开发时间
4. 减少存储和计算资源需求

**实施方式**：
```python
# config/file_filters.py
SUPPORTED_FORMATS = {
    'images': ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.heic'],
    'videos': []  # Phase 2启用: ['.mp4', '.mov', '.avi', '.mkv']
}

def should_process_file(file_path: Path) -> bool:
    """MVP阶段仅处理图片"""
    return file_path.suffix.lower() in SUPPORTED_FORMATS['images']
```

#### 方案B：Phase 2引入视频处理

**时间线**：MVP后2-3个月

**功能规划**：
1. 关键帧提取
2. 场景检测
3. 内容识别
4. 视频搜索

## 🎬 Phase 2 视频处理架构

### 技术栈选型

#### 核心库
- **FFmpeg**: 视频处理瑞士军刀
- **OpenCV**: 计算机视觉处理
- **PyAV**: Python的FFmpeg绑定
- **scikit-video**: 视频处理工具集

### 视频处理流水线

```
视频文件 → 预处理 → 关键帧提取 → 场景分析 → 内容识别 → 索引存储
   │          │           │            │           │           │
   ▼          ▼           ▼            ▼           ▼           ▼
 验证格式   转码压缩   智能采样    场景切分   物体/人脸   向量+元数据
```

## 🔧 详细实现方案

### 1. 视频预处理模块

```python
# services/video_preprocessor.py
import ffmpeg
from pathlib import Path
import hashlib

class VideoPreprocessor:
    """视频预处理服务"""
    
    def __init__(self):
        self.supported_formats = ['.mp4', '.mov', '.avi', '.mkv', '.wmv']
        self.target_format = 'mp4'
        self.max_resolution = (1920, 1080)
    
    async def process_video(self, video_path: Path):
        """处理单个视频文件"""
        
        # 1. 验证视频文件
        if not await self.validate_video(video_path):
            return None
        
        # 2. 获取视频信息
        metadata = await self.extract_metadata(video_path)
        
        # 3. 计算视频指纹
        video_hash = await self.calculate_video_hash(video_path)
        
        # 4. 检查重复
        if await self.is_duplicate(video_hash):
            return {'status': 'duplicate', 'hash': video_hash}
        
        # 5. 转码优化（如需要）
        if self.needs_transcoding(metadata):
            optimized_path = await self.transcode_video(video_path, metadata)
        else:
            optimized_path = video_path
        
        # 6. 生成预览
        preview_path = await self.generate_preview(optimized_path)
        
        return {
            'status': 'success',
            'video_path': optimized_path,
            'preview_path': preview_path,
            'metadata': metadata,
            'hash': video_hash
        }
    
    async def extract_metadata(self, video_path: Path) -> dict:
        """提取视频元数据"""
        probe = ffmpeg.probe(str(video_path))
        
        video_stream = next(s for s in probe['streams'] 
                          if s['codec_type'] == 'video')
        
        return {
            'duration': float(probe['format']['duration']),
            'width': video_stream['width'],
            'height': video_stream['height'],
            'fps': eval(video_stream['r_frame_rate']),
            'codec': video_stream['codec_name'],
            'bitrate': int(probe['format']['bit_rate']),
            'size': int(probe['format']['size']),
            'creation_time': probe['format'].get('creation_time')
        }
    
    async def transcode_video(self, input_path: Path, metadata: dict) -> Path:
        """视频转码优化"""
        output_path = input_path.with_suffix('.mp4')
        
        # 构建转码参数
        stream = ffmpeg.input(str(input_path))
        
        # 分辨率调整
        if metadata['width'] > self.max_resolution[0]:
            stream = ffmpeg.filter(stream, 'scale', 
                                 w=self.max_resolution[0], 
                                 h=-1)
        
        # H.264编码，Web兼容
        stream = ffmpeg.output(stream, str(output_path),
                             vcodec='libx264',
                             acodec='aac',
                             crf=23,  # 质量因子
                             preset='medium')
        
        # 执行转码
        await ffmpeg.run_async(stream)
        
        return output_path
```

### 2. 关键帧提取模块

```python
# services/keyframe_extractor.py
import cv2
import numpy as np
from typing import List
from sklearn.cluster import KMeans

class KeyframeExtractor:
    """智能关键帧提取"""
    
    def __init__(self):
        self.sampling_rate = 1.0  # 每秒采样帧数
        self.max_keyframes = 20   # 最大关键帧数
        self.min_scene_duration = 2.0  # 最小场景时长（秒）
    
    async def extract_keyframes(self, video_path: str) -> List[np.ndarray]:
        """提取关键帧"""
        
        # 1. 打开视频
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 2. 采样策略
        sample_interval = int(fps / self.sampling_rate)
        sampled_frames = []
        
        frame_idx = 0
        while cap.isOpened() and len(sampled_frames) < 100:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                # 缩小尺寸加速处理
                small_frame = cv2.resize(frame, (224, 224))
                sampled_frames.append(small_frame)
            
            frame_idx += 1
        
        cap.release()
        
        # 3. 场景检测
        scenes = await self.detect_scenes(sampled_frames)
        
        # 4. 关键帧选择
        keyframes = await self.select_keyframes(sampled_frames, scenes)
        
        return keyframes
    
    async def detect_scenes(self, frames: List[np.ndarray]) -> List[int]:
        """检测场景变化"""
        scenes = [0]  # 第一帧是场景开始
        
        for i in range(1, len(frames)):
            # 计算帧差异
            diff = self.calculate_frame_difference(frames[i-1], frames[i])
            
            # 场景切换阈值
            if diff > 0.4:  # 40%差异认为是新场景
                scenes.append(i)
        
        return scenes
    
    def calculate_frame_difference(self, frame1: np.ndarray, 
                                  frame2: np.ndarray) -> float:
        """计算帧间差异"""
        # 转换为灰度图
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 计算直方图
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # 归一化
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        # 计算相关性
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return 1 - correlation
    
    async def select_keyframes(self, frames: List[np.ndarray], 
                              scenes: List[int]) -> List[np.ndarray]:
        """选择代表性关键帧"""
        keyframes = []
        
        for i in range(len(scenes)):
            # 场景范围
            start_idx = scenes[i]
            end_idx = scenes[i+1] if i+1 < len(scenes) else len(frames)
            
            scene_frames = frames[start_idx:end_idx]
            
            if len(scene_frames) == 0:
                continue
            
            # 每个场景选择最有代表性的帧
            if len(scene_frames) == 1:
                keyframes.append(scene_frames[0])
            else:
                # 使用聚类找中心帧
                representative = await self.find_representative_frame(scene_frames)
                keyframes.append(representative)
        
        # 限制关键帧数量
        if len(keyframes) > self.max_keyframes:
            keyframes = self.sample_keyframes(keyframes, self.max_keyframes)
        
        return keyframes
```

### 3. 视频内容识别

```python
# services/video_analyzer.py
from typing import List, Dict
import torch

class VideoContentAnalyzer:
    """视频内容分析器"""
    
    def __init__(self):
        self.frame_detector = RTMDetector()  # 复用图片检测器
        self.action_recognizer = self.load_action_model()
        self.scene_classifier = self.load_scene_model()
    
    async def analyze_video(self, video_path: str, keyframes: List[np.ndarray]):
        """分析视频内容"""
        
        # 1. 分析关键帧
        frame_results = []
        for frame in keyframes:
            detection = await self.analyze_frame(frame)
            frame_results.append(detection)
        
        # 2. 聚合分析结果
        aggregated = self.aggregate_frame_results(frame_results)
        
        # 3. 视频级别分类
        video_category = await self.classify_video_content(aggregated)
        
        # 4. 生成标签
        tags = self.generate_video_tags(aggregated, video_category)
        
        # 5. 提取亮点片段
        highlights = await self.extract_highlights(video_path, frame_results)
        
        return {
            'category': video_category,
            'tags': tags,
            'objects': aggregated['objects'],
            'scenes': aggregated['scenes'],
            'highlights': highlights,
            'frame_analysis': frame_results
        }
    
    async def analyze_frame(self, frame: np.ndarray) -> dict:
        """分析单帧"""
        # 物体检测
        objects = self.frame_detector.detect(frame)
        
        # 场景识别
        scene = self.scene_classifier.classify(frame)
        
        # 文字检测（如有）
        text = await extract_text_from_frame(frame) if self.has_text(frame) else None
        
        return {
            'objects': objects,
            'scene': scene,
            'text': text
        }
    
    def aggregate_frame_results(self, results: List[dict]) -> dict:
        """聚合多帧分析结果"""
        all_objects = {}
        all_scenes = {}
        
        for result in results:
            # 统计物体出现频率
            for obj in result['objects']:
                label = obj['label']
                all_objects[label] = all_objects.get(label, 0) + 1
            
            # 统计场景类型
            scene = result['scene']
            all_scenes[scene] = all_scenes.get(scene, 0) + 1
        
        return {
            'objects': sorted(all_objects.items(), key=lambda x: x[1], reverse=True),
            'scenes': sorted(all_scenes.items(), key=lambda x: x[1], reverse=True),
            'dominant_objects': [k for k, v in all_objects.items() if v > len(results) * 0.3],
            'dominant_scene': max(all_scenes.items(), key=lambda x: x[1])[0]
        }
```

### 4. 视频数据存储

```sql
-- 视频元数据表
CREATE TABLE videos (
    id BIGSERIAL PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    file_hash TEXT UNIQUE NOT NULL,
    
    -- 基础信息
    duration_seconds FLOAT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    fps FLOAT,
    codec TEXT,
    file_size_bytes BIGINT,
    
    -- 处理状态
    processing_status TEXT DEFAULT 'pending',
    processed_at TIMESTAMP,
    error_message TEXT,
    
    -- 分析结果
    category TEXT,
    tags TEXT[],
    dominant_objects TEXT[],
    dominant_scene TEXT,
    
    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 关键帧表
CREATE TABLE video_keyframes (
    id BIGSERIAL PRIMARY KEY,
    video_id BIGINT REFERENCES videos(id) ON DELETE CASCADE,
    
    frame_index INTEGER NOT NULL,
    timestamp_seconds FLOAT NOT NULL,
    thumbnail_path TEXT NOT NULL,
    
    -- 帧分析结果
    detected_objects JSONB,
    scene_type TEXT,
    quality_score FLOAT,
    
    -- 向量嵌入
    embedding vector(768),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(video_id, frame_index)
);

-- 视频片段表（高亮/剪辑）
CREATE TABLE video_segments (
    id BIGSERIAL PRIMARY KEY,
    video_id BIGINT REFERENCES videos(id) ON DELETE CASCADE,
    
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    duration FLOAT GENERATED ALWAYS AS (end_time - start_time) STORED,
    
    segment_type TEXT,  -- 'highlight', 'scene', 'manual'
    description TEXT,
    tags TEXT[],
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 索引优化
CREATE INDEX idx_videos_category ON videos(category);
CREATE INDEX idx_videos_tags ON videos USING GIN(tags);
CREATE INDEX idx_keyframes_video ON video_keyframes(video_id);
CREATE INDEX idx_segments_video ON video_segments(video_id);
```

## 🎯 渐进式实施计划

### Phase 1: MVP（当前）
```yaml
scope:
  included:
    - 图片处理（PNG, JPG, HEIC等）
    - 基础检测和分类
    - 文字提取（OCR）
    - 相似搜索
  
  excluded:
    - 视频处理（762个MOV文件）
    - 动图处理（GIF动画）
    
implementation:
  - 在文件扫描时跳过视频文件
  - 记录视频文件路径供后续处理
  - UI显示视频文件但标记为"待处理"
```

### Phase 2: 视频支持（MVP+2月）
```yaml
features:
  - 关键帧提取和索引
  - 视频内容搜索
  - 视频预览生成
  - 场景检测
  
tech_stack:
  - FFmpeg处理
  - OpenCV分析
  - 继承现有检测器
  
estimated_effort: 3-4周
```

### Phase 3: 高级视频功能（MVP+6月）
```yaml
advanced_features:
  - 动作识别
  - 自动剪辑
  - 视频摘要生成
  - 人物追踪
  - 音频分析
  
tech_requirements:
  - GPU加速
  - 专用视频模型
  - 流媒体服务器
```

## 📊 资源评估

### 存储需求
| 阶段 | 图片 | 视频 | 总计 |
|------|------|------|------|
| MVP | 400GB | 0 | 400GB |
| Phase 2 | 400GB | 100GB原始 + 20GB处理后 | 520GB |
| Phase 3 | 400GB | 100GB原始 + 50GB衍生 | 550GB |

### 计算资源
| 任务 | CPU | 内存 | GPU | 时间/视频 |
|------|-----|------|-----|-----------|
| 转码 | 4核 | 8GB | 可选 | 1-5分钟 |
| 关键帧提取 | 2核 | 4GB | - | 30秒 |
| 内容分析 | 4核 | 8GB | 推荐 | 1-2分钟 |

## 🚦 决策建议

### 推荐方案：分阶段实施

1. **MVP阶段（现在）**：
   - ✅ 专注图片处理，快速验证核心价值
   - ✅ 预留视频处理接口
   - ✅ 收集用户对视频功能的需求反馈

2. **观察期（MVP后1月）**：
   - 📊 分析用户使用数据
   - 🎯 确定视频处理优先级
   - 💡 收集具体视频场景需求

3. **视频功能（按需启动）**：
   - 🎬 基于用户需求决定是否开发
   - 📈 评估ROI（开发成本vs用户价值）
   - 🔧 渐进式添加视频功能

## 📋 实施检查清单

### MVP阶段
- [x] 明确排除视频处理
- [ ] 在文档中说明视频处理计划
- [ ] 文件过滤器排除视频格式
- [ ] UI提示视频文件"即将支持"
- [ ] 记录视频文件元信息

### Phase 2准备
- [ ] 评估用户视频需求
- [ ] 设计视频处理架构
- [ ] 准备视频处理环境
- [ ] 制定迁移计划
- [ ] 性能基准测试

## 总结

基于当前数据集分析（视频仅占2.5%），建议：

1. **MVP阶段明确排除视频处理**，集中资源优化97.5%的图片处理体验
2. **预留扩展接口**，便于未来添加视频支持
3. **Phase 2根据用户反馈决定**是否引入视频功能
4. **技术方案已准备就绪**，可在2-3周内实现基础视频支持

这样既能快速交付MVP，又为未来视频功能预留了清晰的技术路径。
