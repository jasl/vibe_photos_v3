# PoC1 实施计划

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
- [ ] 集成RTMDet或CLIP模型
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
- [ ] 实现批量导入界面
- [ ] 创建搜索和浏览界面
- [ ] 测试和bug修复

## 🛠 具体实现步骤

### Step 1: 项目初始化

```bash
# 创建项目结构
mkdir -p poc1/{app,processors,ui,scripts,tests,data}
mkdir -p poc1/app/api
mkdir -p poc1/data/{images,thumbnails,cache}

# 创建虚拟环境
cd poc1
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装基础依赖（使用最新稳定版本）
pip install fastapi==0.121.1 uvicorn==0.38.0 streamlit==1.51.0 sqlalchemy==2.0.44 pillow==12.0.0 pydantic==2.12.4

# 安装RTMDet依赖（推荐的识别引擎）
pip install torch==2.9.0 torchvision==0.24.0 mmdet==3.3.0 mmengine==0.10.7 mmcv==2.2.0
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

### Step 3: 批处理器实现

```python
# processors/batch.py
import asyncio
from pathlib import Path
from typing import List
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, db_session, detector, ocr_engine):
        self.db = db_session
        self.detector = detector
        self.ocr = ocr_engine
        
    async def process_folder(self, folder_path: str, batch_size: int = 10):
        """批量处理文件夹中的图片"""
        folder = Path(folder_path)
        image_files = list(folder.glob("**/*.jpg")) + \
                     list(folder.glob("**/*.jpeg")) + \
                     list(folder.glob("**/*.png"))
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # 分批处理
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i+batch_size]
            await self._process_batch(batch)
            
            # 显示进度
            progress = min(i + batch_size, len(image_files))
            logger.info(f"Progress: {progress}/{len(image_files)}")
    
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
        """处理单张图片"""
        # 1. 添加到数据库
        image_record = self._add_image_to_db(image_path)
        
        # 2. 生成缩略图
        thumbnail_path = self._create_thumbnail(image_path)
        
        # 3. 物体检测
        detections = await self.detector.detect(image_path)
        self._save_detections(image_record.id, detections)
        
        # 4. OCR（如果需要）
        if self._should_ocr(image_path):
            ocr_result = await self.ocr.extract(image_path)
            self._save_ocr_result(image_record.id, ocr_result)
        
        # 5. 更新状态
        self._update_status(image_record.id, 'completed')
```

### Step 4: 搜索功能实现

```python
# app/api/search.py
from fastapi import APIRouter, Query
from typing import List, Optional
from sqlalchemy import text

router = APIRouter()

@router.get("/search")
async def search_images(
    q: str = Query(..., description="搜索关键词"),
    category: Optional[str] = None,
    min_confidence: float = 0.3,
    limit: int = 50
):
    """搜索图片"""
    
    # 构建搜索查询
    query = """
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
    
    # 关键词搜索
    if q:
        conditions.append(
            "(d.object_class LIKE :keyword OR o.text_content LIKE :keyword)"
        )
        params['keyword'] = f"%{q}%"
    
    # 类别过滤
    if category:
        conditions.append("d.object_class = :category")
        params['category'] = category
    
    # 置信度过滤
    conditions.append("d.confidence >= :min_confidence")
    params['min_confidence'] = min_confidence
    
    if conditions:
        query += " AND " + " AND ".join(conditions)
    
    query += f" LIMIT {limit}"
    
    # 执行查询
    results = db.execute(text(query), params).fetchall()
    
    return {
        "results": [
            {
                "id": r.id,
                "filename": r.filename,
                "thumbnail": r.thumbnail_path,
                "object_class": r.object_class,
                "confidence": r.confidence,
                "ocr_text": r.text_content[:100] if r.text_content else None
            }
            for r in results
        ],
        "total": len(results),
        "query": q
    }
```

### Step 5: Streamlit UI实现

```python
# ui/app.py
import streamlit as st
import requests
from pathlib import Path

st.set_page_config(
    page_title="Vibe Photos PoC1",
    page_icon="📸",
    layout="wide"
)

# 侧边栏
with st.sidebar:
    st.title("📸 Vibe Photos PoC1")
    
    page = st.radio(
        "功能选择",
        ["批量导入", "搜索浏览", "处理状态"]
    )

# 主页面
if page == "批量导入":
    st.header("批量导入图片")
    
    folder_path = st.text_input("图片文件夹路径")
    
    col1, col2 = st.columns(2)
    with col1:
        batch_size = st.number_input("批处理大小", min_value=1, max_value=50, value=10)
    with col2:
        enable_ocr = st.checkbox("启用OCR", value=True)
    
    if st.button("开始导入", type="primary"):
        with st.spinner("处理中..."):
            response = requests.post(
                "http://localhost:8000/batch/import",
                json={
                    "folder_path": folder_path,
                    "batch_size": batch_size,
                    "enable_ocr": enable_ocr
                }
            )
            
            if response.ok:
                st.success("导入任务已启动！")
                st.json(response.json())
            else:
                st.error("导入失败")

elif page == "搜索浏览":
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
- [x] 基础物体检测（RTMDet/CLIP）
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
cd poc1

# 2. 安装依赖
pip install -r requirements.txt

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

→ 查看[测试方案](testing.md)了解如何验证PoC1的效果
