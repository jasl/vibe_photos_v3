# 🤖 AI开发指南 - Vibe Photos智能照片管理系统

> 本文档专为Coding AI编写，提供结构化的任务指令和实现规范

## 📋 项目概述

### 系统定义
- **名称**: Vibe Photos
- **类型**: AI智能照片管理系统
- **目标用户**: 自媒体创作者（产品评测、美食推荐、技术教程）
- **核心价值**: 从海量照片中快速找到所需素材

### 技术约束
- **Python版本**: 3.12（固定）
- **包管理器**: uv（必须使用，禁止pip/poetry/conda）
- **开发方式**: 函数式编程优先，避免不必要的类
- **API框架**: FastAPI（异步优先）
- **错误处理**: 早期返回，guard clauses

## 🎯 核心功能需求

### 必须实现的功能
1. **图像识别**: 识别电子产品、美食、文档、人物、风景
2. **智能搜索**: 支持自然语言查询，如"最近拍的iPhone"
3. **批量处理**: 一次处理1000+张照片，速度>10张/秒
4. **OCR提取**: 识别图片中的文字（中英文）
5. **相似分组**: 自动将相似照片分组
6. **增量更新**: 只处理新增照片

### 不实现的功能
- ❌ 图像编辑
- ❌ 云端存储
- ❌ 社交分享
- ❌ 100%全自动化（保留人工干预）

## 🏗️ 项目结构规范

```
vibe_photos_v3/
├── src/                           # 源代码目录
│   ├── core/                      # 核心功能模块
│   │   ├── detector.py            # 图像检测器
│   │   ├── processor.py           # 图像处理器
│   │   ├── searcher.py            # 搜索引擎
│   │   └── database.py            # 数据库操作
│   ├── models/                    # AI模型封装
│   │   ├── siglip_model.py        # SigLIP模型
│   │   ├── blip_model.py          # BLIP模型
│   │   └── ocr_model.py           # OCR模型
│   ├── api/                       # API接口
│   │   ├── main.py                # FastAPI主应用
│   │   ├── routes/                # 路由定义
│   │   │   ├── import_routes.py   # 导入接口
│   │   │   ├── search_routes.py   # 搜索接口
│   │   │   └── annotation_routes.py # 标注接口
│   │   └── schemas.py             # Pydantic模型
│   ├── utils/                     # 工具函数
│   │   ├── image_utils.py         # 图像处理工具
│   │   ├── cache_manager.py       # 缓存管理
│   │   └── logger.py              # 日志配置
│   └── cli.py                     # 命令行接口
├── tests/                         # 测试目录
│   ├── test_detector.py
│   ├── test_search.py
│   └── fixtures/                  # 测试数据
├── config/                        # 配置文件
│   └── settings.yaml
├── data/                          # 数据存储
├── cache/                         # 缓存目录
├── models/                        # 模型文件
└── pyproject.toml                 # 项目配置
```

## 📝 Phase 1: MVP实现（2周）

### 任务1.1: 环境初始化
```bash
# 执行以下命令初始化项目
uv init
uv add torch==2.9.1 torchvision transformers==4.57.1 pillow fastapi uvicorn typer rich
uv add paddlepaddle paddleocr sqlalchemy pydantic
```

### 任务1.2: 实现核心检测器
创建 `src/core/detector.py`:

```python
from typing import Dict, List, Optional
from pathlib import Path
from transformers import AutoModel, AutoProcessor, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class ImageDetector:
    """图像检测器 - 使用SigLIP+BLIP组合"""
    
    def __init__(self):
        # SigLIP: 多语言分类
        self.siglip_processor = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-224-i18n"
        )
        self.siglip_model = AutoModel.from_pretrained(
            "google/siglip-base-patch16-224-i18n"
        )
        
        # BLIP: 图像描述
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
    
    def detect(
        self, 
        image_path: Path,
        candidate_labels: Optional[List[str]] = None
    ) -> Dict:
        """
        检测图像内容
        
        Args:
            image_path: 图像文件路径
            candidate_labels: 候选标签列表（支持中文）
            
        Returns:
            包含分类结果和描述的字典
        """
        # 早期返回模式
        if not image_path.exists():
            return {"error": "Image file not found"}
            
        image = Image.open(image_path).convert("RGB")
        
        # 默认标签
        if candidate_labels is None:
            candidate_labels = [
                "电子产品", "iPhone", "MacBook", "相机",
                "美食", "披萨", "咖啡", "蛋糕",
                "文档", "截图", "证件",
                "人物", "风景", "建筑"
            ]
        
        # SigLIP分类
        inputs = self.siglip_processor(
            text=candidate_labels, 
            images=image,
            padding=True, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.siglip_model(**inputs)
            logits = outputs.logits_per_image
            probs = torch.sigmoid(logits)
        
        # BLIP描述
        caption_inputs = self.blip_processor(image, return_tensors="pt")
        caption_ids = self.blip_model.generate(**caption_inputs, max_length=50)
        caption = self.blip_processor.decode(caption_ids[0], skip_special_tokens=True)
        
        # 构建结果
        classifications = {
            label: float(prob) 
            for label, prob in zip(candidate_labels, probs[0])
            if prob > 0.1  # 只返回置信度>10%的结果
        }
        
        # 按置信度排序
        sorted_classifications = dict(
            sorted(classifications.items(), key=lambda x: x[1], reverse=True)
        )
        
        return {
            "image_path": str(image_path),
            "classifications": sorted_classifications,
            "top_category": list(sorted_classifications.keys())[0] if sorted_classifications else "unknown",
            "confidence": list(sorted_classifications.values())[0] if sorted_classifications else 0.0,
            "caption": caption,
            "status": "success"
        }
```

### 任务1.3: 实现数据库层
创建 `src/core/database.py`:

```python
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Optional, List, Dict
import json

Base = declarative_base()

class Photo(Base):
    """照片数据模型"""
    __tablename__ = "photos"
    
    id = Column(Integer, primary_key=True)
    path = Column(String, unique=True, nullable=False)
    hash = Column(String, index=True)
    
    # 元数据
    width = Column(Integer)
    height = Column(Integer)
    size = Column(Integer)
    taken_at = Column(DateTime)
    imported_at = Column(DateTime, default=datetime.utcnow)
    
    # AI识别结果
    category = Column(String, index=True)
    confidence = Column(Float)
    classifications = Column(JSON)  # 所有分类结果
    caption = Column(Text)  # BLIP生成的描述
    ocr_text = Column(Text)  # OCR提取的文字
    
    # 用户数据
    user_label = Column(String, index=True)
    user_tags = Column(Text)  # 逗号分隔的标签
    
    # 向量嵌入（为Phase 2预留）
    embedding_json = Column(Text)  # JSON序列化的向量

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: str = "data/vibe_photos.db"):
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def add_photo(self, photo_data: Dict) -> int:
        """添加照片记录"""
        session = self.Session()
        try:
            photo = Photo(**photo_data)
            session.add(photo)
            session.commit()
            return photo.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def search_photos(
        self, 
        query: str,
        category: Optional[str] = None,
        min_confidence: float = 0.5,
        limit: int = 50
    ) -> List[Photo]:
        """搜索照片"""
        session = self.Session()
        try:
            q = session.query(Photo)
            
            # 文本搜索
            if query:
                search_pattern = f"%{query}%"
                q = q.filter(
                    (Photo.category.like(search_pattern)) |
                    (Photo.caption.like(search_pattern)) |
                    (Photo.ocr_text.like(search_pattern)) |
                    (Photo.user_label.like(search_pattern)) |
                    (Photo.user_tags.like(search_pattern))
                )
            
            # 类别过滤
            if category:
                q = q.filter(Photo.category == category)
            
            # 置信度过滤
            q = q.filter(Photo.confidence >= min_confidence)
            
            # 排序和限制
            q = q.order_by(Photo.confidence.desc()).limit(limit)
            
            return q.all()
        finally:
            session.close()
    
    def get_photo_by_path(self, path: str) -> Optional[Photo]:
        """根据路径获取照片"""
        session = self.Session()
        try:
            return session.query(Photo).filter_by(path=path).first()
        finally:
            session.close()
    
    def update_photo(self, photo_id: int, updates: Dict) -> bool:
        """更新照片信息"""
        session = self.Session()
        try:
            photo = session.query(Photo).filter_by(id=photo_id).first()
            if not photo:
                return False
            
            for key, value in updates.items():
                if hasattr(photo, key):
                    setattr(photo, key, value)
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
```

### 任务1.4: 实现批处理器
创建 `src/core/processor.py`:

```python
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import hashlib
from PIL import Image
import logging

from .detector import ImageDetector
from .database import DatabaseManager

logger = logging.getLogger(__name__)

class BatchProcessor:
    """批量图像处理器"""
    
    def __init__(
        self, 
        detector: Optional[ImageDetector] = None,
        db_manager: Optional[DatabaseManager] = None,
        cache_dir: Path = Path("cache")
    ):
        self.detector = detector or ImageDetector()
        self.db = db_manager or DatabaseManager()
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_hash(self, image_path: Path) -> str:
        """计算图像哈希值用于去重"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def extract_metadata(self, image_path: Path) -> Dict:
        """提取图像元数据"""
        try:
            img = Image.open(image_path)
            return {
                "width": img.width,
                "height": img.height,
                "size": image_path.stat().st_size,
            }
        except Exception as e:
            logger.error(f"Failed to extract metadata from {image_path}: {e}")
            return {}
    
    def process_single_image(self, image_path: Path) -> Dict:
        """处理单张图像"""
        # 检查是否已处理
        existing = self.db.get_photo_by_path(str(image_path))
        if existing:
            logger.info(f"Skipping already processed: {image_path}")
            return {"status": "skipped", "path": str(image_path)}
        
        try:
            # 计算哈希
            image_hash = self.compute_hash(image_path)
            
            # 提取元数据
            metadata = self.extract_metadata(image_path)
            
            # AI检测
            detection_result = self.detector.detect(image_path)
            
            # 准备数据库记录
            photo_data = {
                "path": str(image_path),
                "hash": image_hash,
                **metadata,
                "category": detection_result.get("top_category"),
                "confidence": detection_result.get("confidence"),
                "classifications": detection_result.get("classifications"),
                "caption": detection_result.get("caption"),
            }
            
            # 保存到数据库
            photo_id = self.db.add_photo(photo_data)
            
            return {
                "status": "success",
                "path": str(image_path),
                "photo_id": photo_id,
                "category": photo_data["category"],
                "confidence": photo_data["confidence"]
            }
            
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            return {
                "status": "error",
                "path": str(image_path),
                "error": str(e)
            }
    
    def process_batch(
        self, 
        directory: Path,
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.webp'],
        max_workers: int = 4
    ) -> Dict:
        """批量处理目录中的图像"""
        # 收集图像文件
        image_files = []
        for ext in extensions:
            image_files.extend(directory.glob(f"**/*{ext}"))
            image_files.extend(directory.glob(f"**/*{ext.upper()}"))
        
        if not image_files:
            return {
                "status": "no_images",
                "message": f"No images found in {directory}"
            }
        
        results = {
            "total": len(image_files),
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "details": []
        }
        
        # 并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_single_image, img): img 
                for img in image_files
            }
            
            # 显示进度条
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                result = future.result()
                results["details"].append(result)
                
                if result["status"] == "success":
                    results["processed"] += 1
                elif result["status"] == "skipped":
                    results["skipped"] += 1
                else:
                    results["errors"] += 1
        
        return results
```

### 任务1.5: 实现OCR功能
创建 `src/models/ocr_model.py`:

```python
from paddleocr import PaddleOCR
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class OCRExtractor:
    """OCR文字提取器"""
    
    def __init__(self, lang: str = "ch"):
        """
        初始化OCR模型
        
        Args:
            lang: 语言设置，'ch'表示中英文混合，'en'表示纯英文
        """
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=False,  # CPU模式
            show_log=False
        )
    
    def extract_text(self, image_path: Path) -> Dict:
        """
        从图像中提取文字
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            包含提取文字和置信度的字典
        """
        if not image_path.exists():
            return {"error": "Image file not found", "text": ""}
        
        try:
            # OCR识别
            result = self.ocr.ocr(str(image_path), cls=True)
            
            if not result or not result[0]:
                return {"text": "", "confidence": 0.0, "lines": []}
            
            # 提取文字行
            lines = []
            all_text = []
            total_confidence = 0.0
            
            for line in result[0]:
                text = line[1][0]
                confidence = line[1][1]
                
                lines.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": line[0]
                })
                
                all_text.append(text)
                total_confidence += confidence
            
            # 计算平均置信度
            avg_confidence = total_confidence / len(lines) if lines else 0.0
            
            return {
                "text": " ".join(all_text),
                "confidence": avg_confidence,
                "lines": lines,
                "line_count": len(lines)
            }
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            return {
                "error": str(e),
                "text": "",
                "confidence": 0.0,
                "lines": []
            }
    
    def is_document(self, ocr_result: Dict, min_lines: int = 5) -> bool:
        """
        判断图像是否为文档类型
        
        Args:
            ocr_result: OCR提取结果
            min_lines: 最少文字行数阈值
            
        Returns:
            是否为文档
        """
        return ocr_result.get("line_count", 0) >= min_lines
```

### 任务1.6: 实现FastAPI接口
创建 `src/api/main.py`:

```python
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from pathlib import Path
import asyncio

from ..core.detector import ImageDetector
from ..core.database import DatabaseManager
from ..core.processor import BatchProcessor
from ..models.ocr_model import OCRExtractor

app = FastAPI(title="Vibe Photos API", version="1.0.0")

# 初始化组件
detector = ImageDetector()
db_manager = DatabaseManager()
processor = BatchProcessor(detector, db_manager)
ocr = OCRExtractor()

class ImportRequest(BaseModel):
    directory: str
    extensions: List[str] = ['.jpg', '.jpeg', '.png']
    max_workers: int = 4

class SearchRequest(BaseModel):
    query: str
    category: Optional[str] = None
    min_confidence: float = 0.5
    limit: int = 50

class AnnotationRequest(BaseModel):
    photo_id: int
    user_label: str
    user_tags: Optional[List[str]] = []

@app.get("/")
async def root():
    """API根路径"""
    return {
        "name": "Vibe Photos API",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/import/batch")
async def import_batch(request: ImportRequest):
    """批量导入照片"""
    directory = Path(request.directory)
    
    if not directory.exists():
        raise HTTPException(status_code=400, detail="Directory not found")
    
    # 异步执行批处理
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        processor.process_batch,
        directory,
        request.extensions,
        request.max_workers
    )
    
    return JSONResponse(content=result)

@app.post("/import/single")
async def import_single(file: UploadFile = File(...)):
    """导入单张照片"""
    # 保存上传文件
    temp_path = Path(f"temp/{file.filename}")
    temp_path.parent.mkdir(exist_ok=True)
    
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # 处理图像
    result = processor.process_single_image(temp_path)
    
    # 清理临时文件
    temp_path.unlink()
    
    return result

@app.post("/search")
async def search_photos(request: SearchRequest):
    """搜索照片"""
    photos = db_manager.search_photos(
        query=request.query,
        category=request.category,
        min_confidence=request.min_confidence,
        limit=request.limit
    )
    
    results = []
    for photo in photos:
        results.append({
            "id": photo.id,
            "path": photo.path,
            "category": photo.category,
            "confidence": photo.confidence,
            "caption": photo.caption,
            "user_label": photo.user_label
        })
    
    return {
        "query": request.query,
        "count": len(results),
        "results": results
    }

@app.post("/annotate")
async def annotate_photo(request: AnnotationRequest):
    """标注照片"""
    updates = {
        "user_label": request.user_label,
        "user_tags": ",".join(request.user_tags) if request.user_tags else ""
    }
    
    success = db_manager.update_photo(request.photo_id, updates)
    
    if not success:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    return {"status": "success", "photo_id": request.photo_id}

@app.post("/ocr/extract")
async def extract_text(photo_id: int):
    """提取照片中的文字"""
    # 获取照片信息
    photo = db_manager.Session().query(db_manager.Photo).filter_by(id=photo_id).first()
    
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    
    # 执行OCR
    ocr_result = ocr.extract_text(Path(photo.path))
    
    # 更新数据库
    if ocr_result.get("text"):
        db_manager.update_photo(photo_id, {"ocr_text": ocr_result["text"]})
    
    return ocr_result

@app.get("/stats")
async def get_statistics():
    """获取系统统计信息"""
    session = db_manager.Session()
    try:
        from sqlalchemy import func
        
        total_photos = session.query(func.count(db_manager.Photo.id)).scalar()
        
        category_stats = session.query(
            db_manager.Photo.category,
            func.count(db_manager.Photo.id).label("count"),
            func.avg(db_manager.Photo.confidence).label("avg_confidence")
        ).group_by(db_manager.Photo.category).all()
        
        return {
            "total_photos": total_photos,
            "categories": [
                {
                    "name": stat[0],
                    "count": stat[1],
                    "avg_confidence": round(stat[2], 2) if stat[2] else 0
                }
                for stat in category_stats
            ]
        }
    finally:
        session.close()
```

### 任务1.7: 实现命令行接口
创建 `src/cli.py`:

```python
import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track
import json

from .core.detector import ImageDetector
from .core.database import DatabaseManager
from .core.processor import BatchProcessor
from .models.ocr_model import OCRExtractor

app = typer.Typer(help="Vibe Photos CLI - AI照片管理工具")
console = Console()

@app.command()
def import_photos(
    directory: Path = typer.Argument(..., help="照片目录路径"),
    workers: int = typer.Option(4, help="并行处理线程数"),
    ocr: bool = typer.Option(False, help="是否提取文字")
):
    """批量导入照片"""
    console.print(f"[bold blue]正在导入照片目录: {directory}[/bold blue]")
    
    processor = BatchProcessor()
    result = processor.process_batch(directory, max_workers=workers)
    
    # 显示结果
    console.print(f"\n[green]导入完成![/green]")
    console.print(f"总计: {result['total']} 张")
    console.print(f"成功: {result['processed']} 张")
    console.print(f"跳过: {result['skipped']} 张")
    console.print(f"错误: {result['errors']} 张")
    
    # 如果启用OCR
    if ocr and result['processed'] > 0:
        console.print("\n[yellow]正在提取文字...[/yellow]")
        ocr_extractor = OCRExtractor()
        # OCR处理逻辑...

@app.command()
def search(
    query: str = typer.Argument(..., help="搜索关键词"),
    category: str = typer.Option(None, help="指定类别"),
    limit: int = typer.Option(20, help="结果数量限制")
):
    """搜索照片"""
    db = DatabaseManager()
    results = db.search_photos(query, category=category, limit=limit)
    
    if not results:
        console.print(f"[yellow]未找到匹配的照片[/yellow]")
        return
    
    # 创建表格
    table = Table(title=f"搜索结果: {query}")
    table.add_column("ID", style="cyan")
    table.add_column("路径", style="green")
    table.add_column("类别", style="yellow")
    table.add_column("置信度", style="magenta")
    table.add_column("描述", style="white")
    
    for photo in results:
        table.add_row(
            str(photo.id),
            Path(photo.path).name,
            photo.category or "未知",
            f"{photo.confidence:.1%}" if photo.confidence else "N/A",
            (photo.caption or "")[:50] + "..." if photo.caption and len(photo.caption) > 50 else photo.caption or ""
        )
    
    console.print(table)

@app.command()
def detect(
    image_path: Path = typer.Argument(..., help="图像文件路径"),
    show_all: bool = typer.Option(False, help="显示所有分类结果")
):
    """检测单张照片"""
    if not image_path.exists():
        console.print(f"[red]文件不存在: {image_path}[/red]")
        return
    
    detector = ImageDetector()
    result = detector.detect(image_path)
    
    # 显示结果
    console.print(f"\n[bold]检测结果:[/bold]")
    console.print(f"文件: {image_path}")
    console.print(f"主要类别: [yellow]{result['top_category']}[/yellow]")
    console.print(f"置信度: [green]{result['confidence']:.1%}[/green]")
    console.print(f"描述: {result['caption']}")
    
    if show_all:
        console.print("\n[bold]所有分类:[/bold]")
        for label, score in result['classifications'].items():
            console.print(f"  {label}: {score:.1%}")

@app.command()
def stats():
    """显示统计信息"""
    db = DatabaseManager()
    session = db.Session()
    
    try:
        from sqlalchemy import func
        
        total = session.query(func.count(db.Photo.id)).scalar()
        
        if total == 0:
            console.print("[yellow]数据库中没有照片[/yellow]")
            return
        
        # 类别统计
        category_stats = session.query(
            db.Photo.category,
            func.count(db.Photo.id).label("count")
        ).group_by(db.Photo.category).order_by(func.count(db.Photo.id).desc()).all()
        
        console.print(f"\n[bold]照片统计信息[/bold]")
        console.print(f"总计: {total} 张照片\n")
        
        table = Table(title="类别分布")
        table.add_column("类别", style="cyan")
        table.add_column("数量", style="green")
        table.add_column("占比", style="yellow")
        
        for cat, count in category_stats:
            table.add_row(
                cat or "未分类",
                str(count),
                f"{count/total:.1%}"
            )
        
        console.print(table)
        
    finally:
        session.close()

@app.command()
def export(
    query: str = typer.Argument(..., help="导出条件（搜索词或类别）"),
    output_dir: Path = typer.Argument(..., help="输出目录"),
    format: str = typer.Option("json", help="导出格式: json/csv")
):
    """导出照片数据"""
    db = DatabaseManager()
    photos = db.search_photos(query, limit=10000)
    
    if not photos:
        console.print(f"[yellow]没有找到匹配的照片[/yellow]")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        output_file = output_dir / f"export_{query}.json"
        data = []
        for photo in photos:
            data.append({
                "id": photo.id,
                "path": photo.path,
                "category": photo.category,
                "confidence": photo.confidence,
                "caption": photo.caption,
                "ocr_text": photo.ocr_text,
                "user_label": photo.user_label
            })
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    console.print(f"[green]成功导出 {len(photos)} 张照片到 {output_file}[/green]")

if __name__ == "__main__":
    app()
```

### 任务1.8: 创建测试文件
创建 `tests/test_detector.py`:

```python
import pytest
from pathlib import Path
from src.core.detector import ImageDetector
from src.core.database import DatabaseManager
from src.core.processor import BatchProcessor

@pytest.fixture
def detector():
    """创建检测器实例"""
    return ImageDetector()

@pytest.fixture
def db_manager():
    """创建测试数据库"""
    return DatabaseManager("data/test.db")

def test_detector_initialization(detector):
    """测试检测器初始化"""
    assert detector.siglip_model is not None
    assert detector.blip_model is not None

def test_image_detection(detector):
    """测试图像检测"""
    # 假设有测试图像
    test_image = Path("tests/fixtures/test_iphone.jpg")
    if test_image.exists():
        result = detector.detect(test_image)
        
        assert result["status"] == "success"
        assert "classifications" in result
        assert "caption" in result
        assert result["confidence"] > 0

def test_database_operations(db_manager):
    """测试数据库操作"""
    # 添加照片
    photo_data = {
        "path": "/test/photo.jpg",
        "category": "电子产品",
        "confidence": 0.95,
        "caption": "A smartphone on a desk"
    }
    
    photo_id = db_manager.add_photo(photo_data)
    assert photo_id > 0
    
    # 搜索照片
    results = db_manager.search_photos("电子产品")
    assert len(results) > 0
    
    # 更新照片
    success = db_manager.update_photo(photo_id, {"user_label": "iPhone 15"})
    assert success

def test_batch_processing(detector, db_manager):
    """测试批处理"""
    processor = BatchProcessor(detector, db_manager)
    test_dir = Path("tests/fixtures")
    
    if test_dir.exists():
        result = processor.process_batch(test_dir, max_workers=2)
        
        assert result["total"] >= 0
        assert "processed" in result
        assert "errors" in result

# 性能测试
def test_detection_performance(detector):
    """测试检测性能"""
    import time
    
    test_image = Path("tests/fixtures/test_iphone.jpg")
    if test_image.exists():
        start = time.time()
        result = detector.detect(test_image)
        elapsed = time.time() - start
        
        # 应该在2秒内完成
        assert elapsed < 2.0
        assert result["status"] == "success"
```

## 📋 Phase 2: 语义搜索增强（1个月）

### 任务2.1: 实现向量嵌入
创建 `src/models/embedder.py`:

```python
from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image
from pathlib import Path
from typing import List, Union
import numpy as np

class ImageEmbedder:
    """图像向量嵌入器"""
    
    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def encode_image(self, image_path: Path) -> np.ndarray:
        """编码单张图像为向量"""
        image = Image.open(image_path).convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            # 归一化
            embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
        
        return embeddings.numpy().squeeze()
    
    def encode_text(self, text: str) -> np.ndarray:
        """编码文本查询为向量"""
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
        
        return embeddings.numpy().squeeze()
    
    def compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        return float(np.dot(vec1, vec2))
```

### 任务2.2: 实现混合搜索
创建 `src/core/hybrid_searcher.py`:

```python
import numpy as np
from typing import List, Dict, Optional
import json
from pathlib import Path

from .database import DatabaseManager, Photo
from ..models.embedder import ImageEmbedder

class HybridSearcher:
    """混合搜索引擎 - 结合文本和语义搜索"""
    
    def __init__(self, db_manager: DatabaseManager, embedder: Optional[ImageEmbedder] = None):
        self.db = db_manager
        self.embedder = embedder or ImageEmbedder()
        
    def search(
        self,
        query: str,
        mode: str = "hybrid",  # text, vector, hybrid
        limit: int = 50,
        alpha: float = 0.5  # 文本权重 vs 向量权重
    ) -> List[Dict]:
        """
        执行搜索
        
        Args:
            query: 搜索查询
            mode: 搜索模式
            limit: 结果限制
            alpha: 混合搜索中文本搜索的权重（0-1）
        """
        
        if mode == "text":
            return self._text_search(query, limit)
        elif mode == "vector":
            return self._vector_search(query, limit)
        else:  # hybrid
            text_results = self._text_search(query, limit * 2)
            vector_results = self._vector_search(query, limit * 2)
            return self._merge_results(text_results, vector_results, alpha, limit)
    
    def _text_search(self, query: str, limit: int) -> List[Dict]:
        """纯文本搜索"""
        photos = self.db.search_photos(query, limit=limit)
        
        results = []
        for i, photo in enumerate(photos):
            results.append({
                "id": photo.id,
                "path": photo.path,
                "score": 1.0 - (i / len(photos)),  # 简单排名分数
                "category": photo.category,
                "caption": photo.caption,
                "source": "text"
            })
        
        return results
    
    def _vector_search(self, query: str, limit: int) -> List[Dict]:
        """向量语义搜索"""
        # 编码查询
        query_embedding = self.embedder.encode_text(query)
        
        # 获取所有带向量的照片
        session = self.db.Session()
        try:
            photos = session.query(Photo).filter(
                Photo.embedding_json.isnot(None)
            ).all()
            
            if not photos:
                return []
            
            # 计算相似度
            similarities = []
            for photo in photos:
                try:
                    # 解析存储的向量
                    embedding = np.array(json.loads(photo.embedding_json))
                    similarity = self.embedder.compute_similarity(query_embedding, embedding)
                    similarities.append((photo, similarity))
                except:
                    continue
            
            # 排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 构建结果
            results = []
            for photo, score in similarities[:limit]:
                results.append({
                    "id": photo.id,
                    "path": photo.path,
                    "score": float(score),
                    "category": photo.category,
                    "caption": photo.caption,
                    "source": "vector"
                })
            
            return results
            
        finally:
            session.close()
    
    def _merge_results(
        self,
        text_results: List[Dict],
        vector_results: List[Dict],
        alpha: float,
        limit: int
    ) -> List[Dict]:
        """合并文本和向量搜索结果"""
        # 创建ID到结果的映射
        merged = {}
        
        # 处理文本结果
        for result in text_results:
            photo_id = result["id"]
            if photo_id not in merged:
                merged[photo_id] = result.copy()
                merged[photo_id]["final_score"] = result["score"] * alpha
            else:
                merged[photo_id]["final_score"] += result["score"] * alpha
        
        # 处理向量结果
        for result in vector_results:
            photo_id = result["id"]
            if photo_id not in merged:
                merged[photo_id] = result.copy()
                merged[photo_id]["final_score"] = result["score"] * (1 - alpha)
            else:
                merged[photo_id]["final_score"] += result["score"] * (1 - alpha)
                merged[photo_id]["source"] = "hybrid"
        
        # 排序并返回
        final_results = list(merged.values())
        final_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        return final_results[:limit]
```

## 📋 Phase 3: 生产级系统（3个月）

### 任务3.1: PostgreSQL + pgvector配置
创建 `scripts/setup_postgres.sql`:

```sql
-- 创建数据库
CREATE DATABASE vibe_photos;

-- 连接到数据库
\c vibe_photos;

-- 安装pgvector扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 创建主表
CREATE TABLE photos (
    id SERIAL PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    hash VARCHAR(64),
    
    -- 元数据
    width INTEGER,
    height INTEGER,
    size BIGINT,
    taken_at TIMESTAMP,
    imported_at TIMESTAMP DEFAULT NOW(),
    
    -- AI结果
    category VARCHAR(100),
    confidence REAL,
    classifications JSONB,
    caption TEXT,
    ocr_text TEXT,
    
    -- 向量嵌入 (768维 for SigLIP-base)
    embedding vector(768),
    
    -- 用户数据
    user_label VARCHAR(200),
    user_tags TEXT[],
    is_favorite BOOLEAN DEFAULT FALSE,
    
    -- 索引标记
    indexed_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 创建索引
CREATE INDEX idx_photos_category ON photos(category);
CREATE INDEX idx_photos_confidence ON photos(confidence);
CREATE INDEX idx_photos_user_label ON photos(user_label);
CREATE INDEX idx_photos_taken_at ON photos(taken_at);
CREATE INDEX idx_photos_hash ON photos(hash);

-- 创建向量索引（使用HNSW算法）
CREATE INDEX idx_photos_embedding ON photos 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 创建全文搜索索引
CREATE INDEX idx_photos_text_search ON photos 
USING gin(to_tsvector('simple', 
    COALESCE(caption, '') || ' ' || 
    COALESCE(ocr_text, '') || ' ' || 
    COALESCE(user_label, '')
));

-- 创建标注历史表
CREATE TABLE annotation_history (
    id SERIAL PRIMARY KEY,
    photo_id INTEGER REFERENCES photos(id) ON DELETE CASCADE,
    ai_prediction VARCHAR(100),
    user_correction VARCHAR(100),
    confidence REAL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 创建模型版本表
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    accuracy REAL,
    parameters JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT FALSE
);

-- 创建搜索日志表（用于分析和优化）
CREATE TABLE search_logs (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    mode VARCHAR(20),  -- text/vector/hybrid
    result_count INTEGER,
    response_time_ms INTEGER,
    user_feedback INTEGER,  -- 1-5评分
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 任务3.2: 高级搜索实现
创建 `src/core/advanced_searcher.py`:

```python
from typing import List, Dict, Optional
import asyncpg
import numpy as np
from datetime import datetime, timedelta

class AdvancedSearcher:
    """生产级高级搜索引擎"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        
    async def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        高级搜索with RRF（Reciprocal Rank Fusion）
        
        Args:
            query: 搜索查询
            filters: 过滤条件
            limit: 结果限制
        """
        conn = await asyncpg.connect(self.db_url)
        
        try:
            # 向量搜索
            vector_results = await self._vector_search_pg(conn, query, filters, limit)
            
            # 文本搜索
            text_results = await self._text_search_pg(conn, query, filters, limit)
            
            # RRF融合
            merged = self._rrf_merge(
                [vector_results, text_results],
                k=60  # RRF常数
            )
            
            return merged[:limit]
            
        finally:
            await conn.close()
    
    async def _vector_search_pg(
        self,
        conn: asyncpg.Connection,
        query: str,
        filters: Optional[Dict],
        limit: int
    ) -> List[Dict]:
        """PostgreSQL向量搜索"""
        # 这里需要先将查询编码为向量
        # query_embedding = self.embedder.encode_text(query)
        
        sql = """
            SELECT 
                id, path, category, caption,
                1 - (embedding <=> $1) as similarity
            FROM photos
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT $2
        """
        
        # 实际实现需要传入query_embedding
        rows = await conn.fetch(sql, query_embedding, limit)
        
        return [dict(row) for row in rows]
    
    async def _text_search_pg(
        self,
        conn: asyncpg.Connection,
        query: str,
        filters: Optional[Dict],
        limit: int
    ) -> List[Dict]:
        """PostgreSQL全文搜索"""
        sql = """
            SELECT 
                id, path, category, caption,
                ts_rank_cd(
                    to_tsvector('simple', 
                        COALESCE(caption, '') || ' ' || 
                        COALESCE(ocr_text, '')
                    ),
                    plainto_tsquery('simple', $1)
                ) as rank
            FROM photos
            WHERE 
                to_tsvector('simple', 
                    COALESCE(caption, '') || ' ' || 
                    COALESCE(ocr_text, '')
                ) @@ plainto_tsquery('simple', $1)
            ORDER BY rank DESC
            LIMIT $2
        """
        
        rows = await conn.fetch(sql, query, limit)
        return [dict(row) for row in rows]
    
    def _rrf_merge(self, result_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion算法
        
        Args:
            result_lists: 多个排序结果列表
            k: RRF常数（通常为60）
        """
        scores = {}
        
        for results in result_lists:
            for rank, item in enumerate(results, 1):
                photo_id = item['id']
                if photo_id not in scores:
                    scores[photo_id] = {
                        'item': item,
                        'score': 0
                    }
                # RRF公式: 1 / (k + rank)
                scores[photo_id]['score'] += 1.0 / (k + rank)
        
        # 排序
        sorted_items = sorted(
            scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return [item['item'] for item in sorted_items]
```

## 🧪 测试规范

### 单元测试要求
- 每个核心函数必须有对应的测试
- 测试覆盖率目标: >80%
- 使用pytest框架
- Mock外部依赖（模型、数据库）

### 集成测试要求
- API端到端测试
- 数据库事务测试
- 并发处理测试
- 性能基准测试

### 测试数据准备
```python
# tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def test_images_dir():
    """测试图像目录"""
    return Path("tests/fixtures/images")

@pytest.fixture
def test_db():
    """测试数据库"""
    from src.core.database import DatabaseManager
    return DatabaseManager("data/test.db")

@pytest.fixture
def sample_image_data():
    """示例图像数据"""
    return {
        "iphone": "tests/fixtures/iphone.jpg",
        "document": "tests/fixtures/document.png",
        "food": "tests/fixtures/pizza.jpg"
    }
```

## 📊 性能要求

### Phase 1性能指标
- 图像处理: <2秒/张
- 批处理: >10张/秒（并行）
- 搜索响应: <1秒
- 内存使用: <2GB

### Phase 2性能指标
- 向量编码: <500ms/张
- 混合搜索: <500ms
- 索引更新: <100ms
- 内存使用: <4GB

### Phase 3性能指标
- 并发用户: >100
- QPS: >1000
- P95延迟: <500ms
- 可用性: >99.5%

## 🚀 部署指南

### 开发环境
```bash
# 1. 克隆仓库
git clone <repository>
cd vibe_photos_v3

# 2. 安装依赖
uv venv
uv pip sync requirements.txt

# 3. 运行开发服务器
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 4. 运行CLI
uv run python -m src.cli --help
```

### 生产环境
```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 安装Python依赖
RUN pip install uv && \
    uv pip install -r requirements.txt

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

## ✅ 实施检查清单

### Phase 1检查项（每项必须完成）
- [ ] 环境配置完成（Python 3.12 + uv）
- [ ] SigLIP+BLIP模型加载成功
- [ ] 数据库创建和连接正常
- [ ] 批量导入功能工作
- [ ] 基础搜索功能实现
- [ ] CLI工具可用
- [ ] API接口响应正常
- [ ] 单元测试通过率>80%
- [ ] 处理1000张图片无错误
- [ ] 文档完整

### Phase 2检查项
- [ ] 向量嵌入功能实现
- [ ] 混合搜索工作正常
- [ ] 搜索准确率提升>20%
- [ ] OCR功能集成完成
- [ ] Web UI可访问
- [ ] 性能达到指标

### Phase 3检查项
- [ ] PostgreSQL+pgvector部署
- [ ] 高级搜索算法实现
- [ ] 生产级监控配置
- [ ] 负载测试通过
- [ ] 文档和培训完成

## 📚 参考资源

### 模型文档
- [SigLIP模型](https://huggingface.co/google/siglip-base-patch16-224-i18n)
- [BLIP模型](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

### 技术文档
- [FastAPI文档](https://fastapi.tiangolo.com/)
- [pgvector文档](https://github.com/pgvector/pgvector)
- [uv包管理器](https://github.com/astral-sh/uv)

## 🔧 故障排查

### 常见问题和解决方案

1. **模型下载失败**
   ```bash
   # 手动下载模型
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('google/siglip-base-patch16-224-i18n')"
   ```

2. **内存不足**
   - 减少batch_size
   - 使用模型量化
   - 启用梯度检查点

3. **OCR中文识别问题**
   - 确保PaddleOCR使用'ch'模式
   - 检查字体文件

4. **数据库连接问题**
   - 检查PostgreSQL服务状态
   - 验证连接字符串
   - 确认pgvector扩展已安装

---

## 📌 重要提醒

### 给Coding AI的特别说明

1. **严格遵循Python 3.12和uv**：不要使用pip或其他包管理器
2. **函数式编程优先**：避免不必要的类和继承
3. **早期返回模式**：处理错误和边界情况要尽早返回
4. **异步优先**：使用async/await处理I/O操作
5. **类型注解必须**：所有函数签名必须有类型提示
6. **Pydantic验证**：使用Pydantic模型进行输入验证
7. **错误处理完整**：每个可能失败的操作都要有错误处理
8. **日志记录充分**：关键操作必须记录日志
9. **测试驱动开发**：先写测试，再写实现
10. **文档即代码**：代码注释要清晰完整

### 代码质量要求

```python
# ✅ 好的示例
async def process_image(
    image_path: Path,
    confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """处理单张图像并返回检测结果"""
    # 早期验证
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return {"error": "Image not found", "path": str(image_path)}
    
    try:
        # 处理逻辑
        result = await detect_objects(image_path)
        
        # 过滤低置信度结果
        if result.confidence < confidence_threshold:
            return {"status": "low_confidence", "confidence": result.confidence}
        
        return {"status": "success", "data": result}
        
    except Exception as e:
        logger.exception(f"Processing failed for {image_path}")
        return {"error": str(e), "path": str(image_path)}

# ❌ 避免的示例
def process_image(path):
    result = detect_objects(path)
    if result:
        return result
    else:
        return None
```

---

**文档版本**: 1.0.0
**最后更新**: 2024-11-12
**目标受众**: Coding AI
**项目状态**: 开发中
