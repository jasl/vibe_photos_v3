# 实施指南 - Vibe Photos V3

## 🚀 快速开始路线图

### Week 0: 准备阶段（1-2天）
```bash
# 1. 创建新项目
mkdir vibe-photos-v3
cd vibe-photos-v3

# 2. 初始化环境
uv init
uv add torch transformers pillow fastapi typer rich

# 3. 验证环境
uv run python -c "import torch; print(torch.__version__)"
```

### Week 1: MVP实现（5天）

#### Day 1-2: 核心检测器
```python
# src/detector.py - 基础CLIP检测器（MVP阶段）
from transformers import CLIPModel, CLIPProcessor

class SimpleDetector:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def classify(self, image_path, categories):
        # 实现基础分类
        pass

# src/rtmdet_detector.py - 高精度物体检测（Phase 2）
# 安装: pip install mmdet mmcv
from mmdet.apis import init_detector, inference_detector

class RTMDetector:
    def __init__(self):
        # 使用RTMDet-L，Apache-2.0许可，52.8% mAP精度
        config = 'configs/rtmdet/rtmdet_l_8xb32-300e_coco.py'
        checkpoint = 'checkpoints/rtmdet_l.pth'
        self.model = init_detector(config, checkpoint)
    
    def detect(self, image_path):
        # 高精度物体检测，支持80类COCO物体
        return inference_detector(self.model, image_path)
```

#### Day 3: 数据层
```python
# src/database.py
import sqlite3
from datetime import datetime

def init_database():
    conn = sqlite3.connect('vibe_photos.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            category TEXT,
            confidence REAL,
            created_at TIMESTAMP
        )
    ''')
```

#### Day 4: CLI工具
```python
# src/cli.py
import typer

app = typer.Typer()

@app.command()
def import_photos(path: Path):
    """导入照片"""
    pass

@app.command()
def search(query: str):
    """搜索照片"""
    pass
```

#### Day 5: 测试和优化
```python
# tests/test_detector.py
def test_basic_classification():
    detector = SimpleDetector()
    result = detector.classify("test.jpg", ["电子产品", "美食"])
    assert result.category in ["电子产品", "美食"]
```

### Week 2: 核心功能（5天）

#### 功能清单
- [ ] OCR集成（PaddleOCR）
- [ ] 品牌识别（扩展CLIP）
- [ ] Web UI（Gradio）
- [ ] 批量处理优化
- [ ] 标注助手

### Month 1: 完整系统

#### 里程碑
- [ ] Few-shot学习
- [ ] 高级搜索
- [ ] React前端
- [ ] 性能优化
- [ ] 部署脚本

## 🏗 项目结构

### 推荐目录结构
```
vibe-photos-v3/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── detector.py       # AI检测
│   │   ├── recognizer.py     # 混合识别
│   │   ├── learner.py        # Few-shot学习
│   │   └── ocr.py            # 文字提取
│   ├── data/
│   │   ├── database.py       # 数据库操作
│   │   ├── models.py         # 数据模型
│   │   └── cache.py          # 缓存管理
│   ├── api/
│   │   ├── app.py            # FastAPI主应用
│   │   ├── routes.py         # API路由
│   │   └── schemas.py        # Pydantic模型
│   ├── ui/
│   │   ├── gradio_app.py     # Gradio界面
│   │   └── static/           # 静态资源
│   └── cli/
│       └── main.py           # CLI入口
├── tests/
│   ├── test_detector.py
│   ├── test_api.py
│   └── fixtures/
├── config/
│   ├── default.yaml
│   └── categories.yaml
├── scripts/
│   ├── setup.py              # 初始化脚本
│   ├── import_photos.py      # 批量导入
│   └── benchmark.py          # 性能测试
├── docs/
│   └── API.md
├── pyproject.toml
├── README.md
└── .env.example
```

## 📝 开发规范

### 代码风格
```python
# 1. 使用类型提示
from typing import List, Dict, Optional
from pathlib import Path

def process_image(
    image_path: Path,
    categories: List[str],
    confidence_threshold: float = 0.5
) -> Dict[str, float]:
    """处理单张图片"""
    pass

# 2. 使用Pydantic模型
from pydantic import BaseModel

class PhotoMetadata(BaseModel):
    path: str
    category: str
    confidence: float
    tags: List[str] = []

# 3. 异步优先
async def batch_process(images: List[Path]):
    tasks = [process_image(img) for img in images]
    return await asyncio.gather(*tasks)
```

### 错误处理
```python
# 优雅的错误处理
class DetectorError(Exception):
    """检测器基础异常"""
    pass

class ModelNotFoundError(DetectorError):
    """模型未找到"""
    pass

def safe_detect(image_path: Path):
    try:
        return detector.detect(image_path)
    except FileNotFoundError:
        logger.error(f"Image not found: {image_path}")
        return None
    except DetectorError as e:
        logger.error(f"Detection failed: {e}")
        return {"error": str(e)}
```

## 🧪 测试策略

### 测试金字塔
```
        /\
       /UI\       (10%) - E2E测试
      /----\
     /  API \     (20%) - 集成测试
    /--------\
   /  Unit    \   (70%) - 单元测试
  /____________\
```

### 测试示例
```python
# 单元测试
def test_image_classification():
    detector = SimpleDetector()
    result = detector.classify("fixtures/iphone.jpg", ["手机", "电脑"])
    assert result["手机"] > result["电脑"]

# 集成测试
@pytest.mark.asyncio
async def test_api_search():
    async with AsyncClient(app=app) as client:
        response = await client.get("/search?q=iPhone")
        assert response.status_code == 200
        assert len(response.json()["results"]) > 0

# 性能测试
def test_batch_performance():
    images = list(Path("test_images").glob("*.jpg"))
    start = time.time()
    results = batch_process(images)
    elapsed = time.time() - start
    assert elapsed < len(images) * 0.5  # < 0.5秒/张
```

## 🚦 CI/CD配置

### GitHub Actions
```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install uv
        uv sync
    
    - name: Run tests
      run: |
        uv run pytest tests/
    
    - name: Lint
      run: |
        uv run ruff check src/
```

## 🚀 RTMDet-L部署指南

### 安装MMDetection和RTMDet
```bash
# 1. 安装依赖
uv add torch torchvision
uv add mmcv-full mmdet

# 2. 下载RTMDet配置文件
mkdir -p configs/rtmdet
wget https://github.com/open-mmlab/mmdetection/raw/main/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py \
     -O configs/rtmdet/rtmdet_l_8xb32-300e_coco.py

# 3. 下载预训练权重（Apache-2.0许可）
mkdir -p checkpoints
wget https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth \
     -O checkpoints/rtmdet_l.pth
```

### RTMDet vs YOLO对比
| 特性 | RTMDet-L | YOLOv8 |
|------|----------|---------|
| **许可证** | Apache-2.0 ✅ | AGPL-3.0 ⚠️ |
| **商用** | 免费 | 需付费 |
| **mAP精度** | 52.8% | 50.2% |
| **推理速度** | 快速 | 更快 |
| **社区支持** | OpenMMLab | Ultralytics |

### 使用示例
```python
# 高精度物体检测
from src.rtmdet_detector import RTMDetector

detector = RTMDetector()
results = detector.detect("product_photo.jpg")

# 自媒体内容分析
for obj in results:
    if obj['score'] > 0.7:  # 高置信度物体
        print(f"检测到: {obj['label']} - {obj['score']:.1%}")
        # 自动生成标签: #电子产品 #iPhone等
```

## 📊 监控和日志

### 日志配置
```python
# src/utils/logging.py
import logging
from rich.logging import RichHandler

def setup_logging(level="INFO"):
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    return logging.getLogger("vibe_photos")

# 使用
logger = setup_logging()
logger.info("Processing image", extra={"path": "IMG_001.jpg"})
```

### 性能监控
```python
# src/utils/metrics.py
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"{name} took {elapsed:.2f} seconds")

# 使用
with timer("batch_import"):
    import_photos(photo_dir)
```

## 🎯 关键实施要点

### 1. 渐进式开发
```python
# Version 1: 简单实现
def search_v1(query):
    return db.execute("SELECT * FROM photos WHERE category LIKE ?", f"%{query}%")

# Version 2: 添加相似度
def search_v2(query):
    results = search_v1(query)
    return sorted(results, key=lambda x: similarity(query, x.tags))

# Version 3: 向量搜索
def search_v3(query):
    embedding = encode_query(query)
    return vector_db.search(embedding, top_k=20)
```

### 2. 功能开关
```python
# config/features.yaml
features:
  ocr_enabled: false  # 逐步启用
  brand_detection: false
  few_shot_learning: false

# 代码中
if config.features.ocr_enabled:
    text = extract_text(image)
```

### 3. 性能优化清单
- [ ] 使用缩略图进行初步分类
- [ ] 批量处理图片
- [ ] 缓存模型预测结果
- [ ] 异步I/O操作
- [ ] 连接池管理

## 🚀 部署指南

### 开发环境
```bash
# 1. 克隆代码
git clone <repo>
cd vibe-photos-v3

# 2. 安装依赖
uv sync

# 3. 运行开发服务器
uv run uvicorn src.api.app:app --reload
```

### 生产部署
```bash
# 1. 构建Docker镜像
docker build -t vibe-photos:v3 .

# 2. 运行容器
docker run -d \
  -p 8000:8000 \
  -v /path/to/photos:/photos \
  -v /path/to/data:/data \
  vibe-photos:v3

# 3. 使用systemd（可选）
sudo cp vibe-photos.service /etc/systemd/system/
sudo systemctl enable vibe-photos
sudo systemctl start vibe-photos
```

## 📋 实施检查清单

### Week 1 交付
- [ ] 基础分类工作
- [ ] 数据库创建和连接
- [ ] CLI可以导入照片
- [ ] 简单搜索功能
- [ ] 单元测试通过

### Week 2 交付
- [ ] OCR功能集成
- [ ] Web界面可访问
- [ ] 批量处理优化
- [ ] API文档完成
- [ ] 性能基准测试

### Month 1 交付
- [ ] 完整功能实现
- [ ] 前端界面美观
- [ ] 部署脚本就绪
- [ ] 用户文档完整
- [ ] 性能达标

## 💡 实施建议

1. **先跑通流程**，再优化性能
2. **先本地部署**，再考虑云端
3. **先单用户**，再多用户
4. **先英文**，再多语言
5. **先CPU**，再GPU加速

## 🎁 快速启动模板

```python
# quickstart.py
"""
Vibe Photos V3 - 快速启动模板
直接运行看效果：uv run quickstart.py /path/to/photos
"""

import typer
from pathlib import Path
from transformers import pipeline

def main(photo_dir: Path):
    # 初始化分类器
    classifier = pipeline("zero-shot-image-classification")
    
    # 处理照片
    for image in photo_dir.glob("*.jpg"):
        result = classifier(
            str(image),
            candidate_labels=["电子产品", "美食", "文档", "风景"]
        )
        print(f"{image.name}: {result[0]['label']} ({result[0]['score']:.1%})")

if __name__ == "__main__":
    typer.run(main)
```

---

准备好开始了吗？让我们在新仓库中实现这个设计！ 🚀
