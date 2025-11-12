# Phase 1 实施计划

## 📅 开发时间线（14天）

### Phase 0: 环境准备（Day 1）
- [ ] 创建项目仓库和基础结构
- [ ] 设置Python虚拟环境
- [ ] 安装核心依赖
- [ ] 初始化数据库

### Phase 1: 基础框架（Day 2-4）
- [ ] 实现数据模型和数据库操作
- [ ] 搭建FastAPI基础应用
- [ ] 实现文件扫描和导入功能
- [ ] 创建缩略图生成模块

### Phase 2: 识别引擎集成（Day 5-8）
- [ ] 集成SigLIP+BLIP模型
- [ ] 实现批量检测功能
- [ ] 集成PaddleOCR
- [ ] 保存处理结果到数据库

### Phase 3: 搜索和API（Day 9-11）
- [ ] 实现全文搜索功能
- [ ] 开发搜索API接口
- [ ] 实现图片浏览API
- [ ] 添加过滤和排序功能

### Phase 4: UI和测试（Day 12-14）
- [ ] 开发Streamlit UI
- [ ] 创建搜索和浏览界面
- [ ] 测试和bug修复

## 🛠 具体实现步骤

### Step 1: 项目初始化

#### ⚠️ 重要：统一使用 `uv` 管理Python环境

**本项目必须使用 `uv` 管理所有依赖**
- ✅ 使用 `uv venv` 创建虚拟环境
- ✅ 使用 `uv add/remove` 管理依赖
- ✅ 使用 `uv run` 运行脚本
- ❌ 禁止使用 `pip`, `pip-tools`, `poetry`

```bash
# 安装 uv (首次使用)
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或 brew install uv (macOS)

# 创建项目结构
mkdir -p phase1/{app,processors,ui,scripts,tests,data}
mkdir -p phase1/app/api
mkdir -p phase1/data/{images,thumbnails,cache}

# 创建虚拟环境 (使用 uv)
cd phase1
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装基础依赖（使用 uv）
uv add fastapi==0.121.1 uvicorn==0.38.0 streamlit==1.51.0 sqlalchemy==2.0.44 pillow==11.3.0 pydantic==2.11.10

# 安装SigLIP+BLIP依赖（主要识别引擎）
uv add torch==2.9.1 torchvision==0.24.1 transformers==4.57.1 sentence-transformers==5.1.2

# 或者使用 requirements.txt 批量安装
uv pip sync requirements.txt
```

### Step 2: 数据库模型实现

```python
# app/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Image(Base):
    __tablename__ = "images"
    
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    file_size = Column(Integer)
    import_time = Column(DateTime, default=datetime.utcnow)
    process_status = Column(String, default='pending')
    process_time = Column(Float)
    thumbnail_path = Column(String)

class Detection(Base):
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"))
    object_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float)
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)

class OCRResult(Base):
    __tablename__ = "ocr_results"
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"))
    text_content = Column(Text)
    confidence = Column(Float)
    language = Column(String)
```

### Step 3: 图像预处理和批处理实现

```python
# processors/preprocessor.py
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import imagehash
import numpy as np

class ImagePreprocessor:
    """图像预处理器：格式归一化、缩略图生成、去重（带缓存）"""
    
    def __init__(self, config):
        self.target_format = config.get('target_format', 'JPEG')
        self.thumbnail_size = config.get('thumbnail_size', (512, 512))
        self.thumbnail_quality = config.get('thumbnail_quality', 85)
        
        # 缓存目录（可跨版本复用）
        paths = config.get('paths', {})
        self.processed_dir = Path(paths.get('processed', 'cache/images/processed'))
        self.thumbnail_dir = Path(paths.get('thumbnails', 'cache/images/thumbnails'))
        
        # 哈希缓存（避免重复计算）
        self.hash_cache_file = Path('cache/hashes/phash_cache.json')
        self.processed_hashes = self.load_hash_cache()
    
    def load_hash_cache(self) -> set:
        """加载哈希缓存"""
        if self.hash_cache_file.exists():
            import json
            with open(self.hash_cache_file) as f:
                data = json.load(f)
                return set(data.get('hashes', []))
        return set()
    
    def save_hash_cache(self):
        """保存哈希缓存"""
        import json
        self.hash_cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.hash_cache_file, 'w') as f:
            json.dump({'hashes': list(self.processed_hashes)}, f)
        
    def preprocess(self, image_path: Path) -> Optional[dict]:
        """
        预处理单张图片
        
        Returns:
            处理结果字典，如果是重复图片返回None
        """
        # 1. 计算感知哈希（去重）
        phash = self.compute_phash(image_path)
        if self.is_duplicate(phash):
            return None
        
        # 2. 加载和归一化图像
        normalized = self.normalize_image(image_path)
        
        # 3. 生成缩略图
        thumbnail_path = self.generate_thumbnail(normalized, image_path.stem)
        
        # 4. 保存归一化后的图像（如需要）
        processed_path = self.save_normalized(normalized, image_path.stem)
        
        # 5. 记录哈希值
        self.processed_hashes.add(phash)
        
        return {
            'original_path': str(image_path),
            'processed_path': str(processed_path),
            'thumbnail_path': str(thumbnail_path),
            'phash': str(phash),
            'format': self.target_format,
            'size': normalized.size
        }
    
    def compute_phash(self, image_path: Path, hash_size: int = 8) -> str:
        """计算感知哈希用于去重"""
        img = Image.open(image_path)
        phash = imagehash.phash(img, hash_size=hash_size)
        return str(phash)
    
    def is_duplicate(self, phash: str, threshold: int = 5) -> bool:
        """
        检查是否为重复图片
        
        Args:
            phash: 当前图片的感知哈希
            threshold: 相似度阈值（汉明距离）
        """
        current_hash = imagehash.hex_to_hash(phash)
        for existing_hash_str in self.processed_hashes:
            existing_hash = imagehash.hex_to_hash(existing_hash_str)
            if current_hash - existing_hash < threshold:
                return True
        return False
    
    def normalize_image(self, image_path: Path) -> Image.Image:
        """
        图像格式归一化
        - 转换为RGB模式
        - 自动旋转（基于EXIF）
        - 限制最大尺寸
        """
        img = Image.open(image_path)
        
        # 处理EXIF方向
        try:
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)
        except:
            pass
        
        # 转换为RGB（如果需要）
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        
        # 限制最大尺寸（保持比例）
        max_size = 4096
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        return img
    
    def generate_thumbnail(self, img: Image.Image, name: str) -> Path:
        """生成缩略图"""
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建缩略图
        thumbnail = img.copy()
        thumbnail.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
        
        # 保存
        thumbnail_path = self.thumbnail_dir / f"{name}_thumb.{self.target_format.lower()}"
        save_kwargs = {'format': self.target_format}
        if self.target_format == 'JPEG':
            save_kwargs['quality'] = self.thumbnail_quality
            save_kwargs['optimize'] = True
        
        thumbnail.save(thumbnail_path, **save_kwargs)
        return thumbnail_path
    
    def save_normalized(self, img: Image.Image, name: str) -> Path:
        """保存归一化后的图像"""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        processed_path = self.processed_dir / f"{name}.{self.target_format.lower()}"
        save_kwargs = {'format': self.target_format}
        if self.target_format == 'JPEG':
            save_kwargs['quality'] = 95
            save_kwargs['optimize'] = True
        
        img.save(processed_path, **save_kwargs)
        return processed_path

# processors/batch.py
import asyncio
from pathlib import Path
from typing import List, Set
from PIL import Image
import logging
import json

logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, db_session, detector, ocr_engine, preprocessor, config=None):
        self.db = db_session
        self.detector = detector
        self.ocr = ocr_engine
        self.preprocessor = preprocessor
        
        # 增量处理支持
        self.state_file = Path(config.get('state_file', 'data/processing_state.json'))
        self.processed_files = self.load_state()
        
        # 测试数据集配置
        self.dataset_dir = Path(config.get('dataset_dir', 'samples'))
        self.supported_formats = config.get('formats', ['.jpg', '.jpeg', '.png', '.heic', '.webp'])
    
    def load_state(self) -> Set[str]:
        """加载处理状态（支持增量处理）"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
                return set(state.get('processed_files', []))
        return set()
    
    def save_state(self):
        """保存处理状态"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump({
                'processed_files': list(self.processed_files),
                'last_updated': str(datetime.now())
            }, f, indent=2)
    
    async def process_dataset(self, incremental: bool = True):
        """
        处理测试数据集
        
        Args:
            incremental: 是否增量处理（跳过已处理的文件）
        """
        if not self.dataset_dir.exists():
            logger.error(f"数据集目录不存在: {self.dataset_dir}")
            return
        
        # 收集所有图片文件
        image_files = []
        for fmt in self.supported_formats:
            image_files.extend(self.dataset_dir.glob(f"**/*{fmt}"))
            image_files.extend(self.dataset_dir.glob(f"**/*{fmt.upper()}"))
        
        # 过滤已处理的文件（增量处理）
        if incremental:
            new_files = [f for f in image_files if str(f) not in self.processed_files]
            if not new_files:
                logger.info("没有新文件需要处理")
                return
            logger.info(f"发现 {len(new_files)} 个新文件（共 {len(image_files)} 个文件）")
            image_files = new_files
        else:
            logger.info(f"处理所有 {len(image_files)} 个文件（非增量模式）")
        
        # 批量处理
        batch_size = 10
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i+batch_size]
            await self._process_batch(batch)
            
            # 显示进度
            progress = min(i + batch_size, len(image_files))
            logger.info(f"进度: {progress}/{len(image_files)}")
            
            # 定期保存状态
            if progress % 50 == 0:
                self.save_state()
        
        # 最终保存状态
        self.save_state()
        logger.info("数据集处理完成")
    
    async def _process_batch(self, image_paths: List[Path]):
        """处理一批图片"""
        tasks = []
        for path in image_paths:
            tasks.append(self._process_single_image(path))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for path, result in zip(image_paths, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process {path}: {result}")
    
    async def _process_single_image(self, image_path: Path):
        """处理单张图片（含预处理和缓存）"""
        try:
            # 1. 预处理（去重、归一化、缩略图）
            preprocess_result = self.preprocessor.preprocess(image_path)
            if preprocess_result is None:
                logger.info(f"跳过重复图片: {image_path}")
                return
            
            # 2. 添加到数据库
            image_record = self._add_image_to_db(
                original_path=image_path,
                processed_path=preprocess_result['processed_path'],
                thumbnail_path=preprocess_result['thumbnail_path'],
                phash=preprocess_result['phash']
            )
            
            # 3. 物体检测（带缓存）
            cache_key = preprocess_result['phash']  # 使用感知哈希作为缓存键
            detections = await self._get_or_compute_detections(
                preprocess_result['processed_path'],
                cache_key
            )
            self._save_detections(image_record.id, detections)
            
            # 4. OCR（带缓存）
            if self._should_ocr(image_path):
                ocr_result = await self._get_or_compute_ocr(
                    preprocess_result['processed_path'],
                    cache_key
                )
                self._save_ocr_result(image_record.id, ocr_result)
            
            # 5. 更新状态
            self._update_status(image_record.id, 'completed')
            
            # 6. 记录已处理
            self.processed_files.add(str(image_path))
            
        except Exception as e:
            logger.error(f"处理图片失败 {image_path}: {e}")
            self._update_status(image_record.id, 'failed', error=str(e))
    
    async def _get_or_compute_detections(self, image_path: str, cache_key: str):
        """获取或计算检测结果（带缓存）"""
        import json
        
        # 缓存文件路径
        cache_dir = Path('cache/detections')
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{cache_key}.json"
        
        # 尝试读取缓存
        if cache_file.exists():
            with open(cache_file) as f:
                logger.debug(f"使用缓存的检测结果: {cache_key}")
                return json.load(f)
        
        # 计算并缓存
        detections = await self.detector.detect(image_path)
        with open(cache_file, 'w') as f:
            json.dump(detections, f)
        
        return detections
    
    async def _get_or_compute_ocr(self, image_path: str, cache_key: str):
        """获取或计算OCR结果（带缓存）"""
        import json
        
        # 缓存文件路径
        cache_dir = Path('cache/ocr')
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{cache_key}.json"
        
        # 尝试读取缓存
        if cache_file.exists():
            with open(cache_file) as f:
                logger.debug(f"使用缓存的OCR结果: {cache_key}")
                return json.load(f)
        
        # 计算并缓存
        ocr_result = await self.ocr.extract(image_path)
        with open(cache_file, 'w') as f:
            json.dump(ocr_result, f)
        
        return ocr_result
```

### Step 4: 搜索功能实现（支持未来扩展）

```python
# app/api/search.py
from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional, Dict, Any
from sqlalchemy import text
import json
import numpy as np

router = APIRouter()

class SearchEngine:
    """搜索引擎，支持渐进式升级"""
    
    def __init__(self, db_session):
        self.db = db_session
        self.vector_enabled = False  # Phase 2开关
    
    async def search(
        self,
        query: str,
        mode: str = "text",
        category: Optional[str] = None,
        min_confidence: float = 0.3,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        统一搜索接口
        
        Args:
            query: 搜索查询
            mode: 搜索模式 ('text', 'vector', 'hybrid')
            category: 类别过滤
            min_confidence: 最低置信度
            limit: 结果限制
        """
        if mode == "text":
            return await self._text_search(query, category, min_confidence, limit)
        elif mode == "vector" and self.vector_enabled:
            return await self._vector_search(query, limit)
        elif mode == "hybrid" and self.vector_enabled:
            return await self._hybrid_search(query, category, min_confidence, limit)
        else:
            raise HTTPException(
                status_code=501, 
                detail=f"搜索模式 '{mode}' 暂未实现（将在Phase 2支持）"
            )
    
    async def _text_search(
        self, 
        query: str, 
        category: Optional[str],
        min_confidence: float,
        limit: int
    ) -> Dict[str, Any]:
        """Phase 1: 文本搜索实现"""
        sql_query = """
            SELECT DISTINCT 
                i.id, i.filename, i.filepath, i.thumbnail_path,
                d.object_class, d.confidence,
                o.text_content
            FROM images i
            LEFT JOIN detections d ON i.id = d.image_id
            LEFT JOIN ocr_results o ON i.id = o.image_id
            WHERE i.process_status = 'completed'
        """
        
        conditions = []
        params = {}
        
        if query:
            conditions.append(
                "(d.object_class LIKE :keyword OR o.text_content LIKE :keyword)"
            )
            params['keyword'] = f"%{query}%"
        
        if category:
            conditions.append("d.object_class = :category")
            params['category'] = category
        
        conditions.append("d.confidence >= :min_confidence")
        params['min_confidence'] = min_confidence
        
        if conditions:
            sql_query += " AND " + " AND ".join(conditions)
        
        sql_query += f" LIMIT {limit}"
        
        results = self.db.execute(text(sql_query), params).fetchall()
        
        return {
            "results": self._format_results(results),
            "total": len(results),
            "mode": "text",
            "query": query
        }
    
    async def _vector_search(self, query: str, limit: int) -> Dict[str, Any]:
        """Phase 2: 向量搜索（预留实现）"""
        # 1. 编码查询
        # query_embedding = self.encode_query(query)
        
        # 2. 从数据库加载向量
        # sql = "SELECT id, embedding_json FROM images WHERE embedding_json IS NOT NULL"
        # images = self.db.execute(text(sql)).fetchall()
        
        # 3. 计算相似度
        # similarities = []
        # for img in images:
        #     embedding = json.loads(img.embedding_json)
        #     sim = cosine_similarity(query_embedding, embedding)
        #     similarities.append((img.id, sim))
        
        # 4. 排序返回
        # similarities.sort(key=lambda x: x[1], reverse=True)
        # top_ids = [s[0] for s in similarities[:limit]]
        
        raise NotImplementedError("向量搜索将在Phase 2实现")
    
    async def _hybrid_search(
        self,
        query: str,
        category: Optional[str],
        min_confidence: float,
        limit: int
    ) -> Dict[str, Any]:
        """Phase 2: 混合搜索（预留实现）"""
        # 1. 并行执行两种搜索
        # text_results = await self._text_search(query, category, min_confidence, limit)
        # vector_results = await self._vector_search(query, limit)
        
        # 2. 融合结果
        # merged = self._merge_results(
        #     text_results['results'],
        #     vector_results['results'],
        #     alpha=0.5
        # )
        
        raise NotImplementedError("混合搜索将在Phase 2实现")
    
    def _format_results(self, raw_results) -> List[Dict]:
        """格式化搜索结果"""
        return [
            {
                "id": r.id,
                "filename": r.filename,
                "thumbnail": r.thumbnail_path,
                "object_class": r.object_class,
                "confidence": r.confidence,
                "ocr_text": r.text_content[:100] if r.text_content else None
            }
            for r in raw_results
        ]
    
    def _merge_results(
        self,
        text_results: List[Dict],
        vector_results: List[Dict],
        alpha: float = 0.5
    ) -> List[Dict]:
        """
        简单的结果融合（Phase 2实现）
        使用加权平均而非复杂的RRF
        """
        # 实现将在Phase 2完成
        pass

# API端点
search_engine = SearchEngine(db_session)

@router.get("/search")
async def search_images(
    q: str = Query(..., description="搜索关键词"),
    mode: str = Query("text", description="搜索模式: text/vector/hybrid"),
    category: Optional[str] = None,
    min_confidence: float = 0.3,
    limit: int = 50
):
    """
    图片搜索API
    
    支持三种模式：
    - text: 文本搜索（Phase 1）
    - vector: 向量搜索（Phase 2）
    - hybrid: 混合搜索（Phase 2）
    """
    return await search_engine.search(
        query=q,
        mode=mode,
        category=category,
        min_confidence=min_confidence,
        limit=limit
    )
```

### Step 5: Streamlit UI实现

```python
# ui/app.py
import streamlit as st
import requests
from pathlib import Path

st.set_page_config(
    page_title="Vibe Photos Phase 1",
    page_icon="📸",
    layout="wide"
)

# 侧边栏
with st.sidebar:
    st.title("📸 Vibe Photos Phase 1")
    
    page = st.radio(
        "功能选择",
        ["搜索浏览", "处理状态"]
    )

# 主页面

if page == "搜索浏览":
    st.header("搜索和浏览")
    
    # 搜索栏
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_query = st.text_input("搜索关键词", placeholder="输入物品名称或文字...")
    with col2:
        category_filter = st.selectbox(
            "类别过滤",
            ["全部", "电子产品", "美食", "文档", "其他"]
        )
    with col3:
        min_confidence = st.slider("最低置信度", 0.0, 1.0, 0.3)
    
    if search_query:
        # 调用搜索API
        response = requests.get(
            "http://localhost:8000/search",
            params={
                "q": search_query,
                "category": None if category_filter == "全部" else category_filter,
                "min_confidence": min_confidence
            }
        )
        
        if response.ok:
            data = response.json()
            st.info(f"找到 {data['total']} 个结果")
            
            # 显示结果网格
            cols = st.columns(4)
            for idx, result in enumerate(data['results']):
                with cols[idx % 4]:
                    if result['thumbnail']:
                        st.image(result['thumbnail'])
                    st.caption(f"{result['filename']}")
                    st.text(f"🏷 {result['object_class']}")
                    st.text(f"📊 {result['confidence']:.1%}")
                    if result['ocr_text']:
                        st.text(f"📝 {result['ocr_text'][:50]}...")

elif page == "处理状态":
    st.header("处理状态监控")
    
    # 获取统计信息
    response = requests.get("http://localhost:8000/stats")
    if response.ok:
        stats = response.json()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总图片数", stats['total_images'])
        with col2:
            st.metric("已处理", stats['processed'], f"+{stats['processing']}")
        with col3:
            st.metric("待处理", stats['pending'])
        with col4:
            st.metric("失败", stats['failed'])
        
        # 显示处理进度
        if stats['total_images'] > 0:
            progress = stats['processed'] / stats['total_images']
            st.progress(progress, text=f"处理进度：{progress:.1%}")
        
        # 刷新按钮
        if st.button("刷新状态"):
            st.rerun()
```

## 📋 核心功能清单

### 必须实现（MVP）
- [x] 批量图片导入
- [x] 缩略图生成
- [x] 基础图像理解（SigLIP+BLIP）
- [x] 简单OCR提取
- [x] SQLite存储
- [x] 关键词搜索
- [x] Web UI界面

### 可选功能（如时间允许）
- [ ] 批处理进度条
- [ ] 导出功能
- [ ] 简单的统计分析
- [ ] 错误重试机制
- [ ] 基础的去重功能

### 不实现（超出范围）
- ❌ 实时处理
- ❌ 向量搜索
- ❌ 用户认证
- ❌ Few-shot学习
- ❌ 复杂的UI交互

## 🧪 测试计划

### 功能测试
1. **导入测试**：测试100张混合类型图片的导入
2. **识别测试**：验证各类物体的识别准确率
3. **OCR测试**：测试中英文文本提取
4. **搜索测试**：测试各种搜索场景

### 性能测试
1. **批处理速度**：测量处理100张图片的时间
2. **搜索响应**：测试1000张图片时的搜索速度
3. **内存占用**：监控处理过程中的内存使用

## 🚀 快速启动指南

```bash
# 1. 克隆代码
git clone <repo-url>
cd phase1

# 2. 安装依赖
uv pip install -r requirements.txt

# 3. 初始化数据库
python scripts/init_db.py

# 4. 启动后端服务
uvicorn app.main:app --reload --port 8000

# 5. 启动前端UI（新终端）
streamlit run ui/app.py --server.port 8501

# 6. 访问界面
# API: http://localhost:8000/docs
# UI: http://localhost:8501
```

## ⚠️ 风险和缓解措施

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 模型加载慢 | 启动时间长 | 使用轻量模型，懒加载 |
| OCR准确率低 | 搜索效果差 | 提供手动编辑功能 |
| 批处理失败 | 数据不完整 | 添加重试机制和日志 |
| UI响应慢 | 用户体验差 | 分页显示，缓存结果 |

## 📝 开发注意事项

1. **保持简单** - 不要过度设计
2. **快速迭代** - 先让它工作，再优化
3. **充分日志** - 记录关键操作便于调试
4. **早期测试** - 每完成一个模块就测试
5. **文档同步** - 及时更新README

## 下一步

→ 查看[测试方案](testing.md)了解如何验证Phase 1的效果
