# 项目依赖说明

## Python 版本要求

**指定版本**: Python 3.12（通过 uv 管理）

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 设置 Python 版本
uv python pin 3.12
```

## 核心依赖版本

### Phase 1 - 核心功能实现

| 分类 | 依赖 | 版本 | 说明 |
|------|------|------|------|
| **Web框架** | FastAPI | 0.121.1 | API服务 |
| | Uvicorn | 0.38.0 | ASGI服务器 |
| | Streamlit | 1.51.0 | UI界面 |
| **AI模型** | PyTorch | 2.9.0 | 深度学习框架 |
| | Transformers | 4.57.1 | SigLIP/BLIP模型 |
| | Sentence-Transformers | 5.1.2 | 语义搜索 |
| **OCR** | PaddlePaddle | 3.2.0 | OCR框架 |
| | PaddleOCR | 3.3.1 | 文字识别 |
| **数据处理** | SQLAlchemy | 2.0.44 | 数据库ORM |
| | Pydantic | 2.11.10 | 数据验证 |
| | NumPy | 2.3.4 | 数值计算 |
| | Pillow | 11.3.0 | 图像处理 |

### Phase Final - 扩展功能

在Phase 1基础上的可选扩展：

| 分类 | 依赖 | 版本 | 使用场景 |
|------|------|------|----------|
| **向量搜索** | pgvector | 0.2.5 | 大规模向量搜索 |
| | psycopg2-binary | 2.9.9 | PostgreSQL连接 |
| **备选UI** | Gradio | 5.49.1 | 交互式界面 |
| **监控** | Loguru | 0.7.3 | 日志管理 |

## 安装指南

### 快速开始（Phase 1）

```bash
cd blueprints/phase1
uv venv --python 3.12
source .venv/bin/activate  # Linux/Mac
uv pip sync requirements.txt
```

### GPU 环境配置

```bash
# NVIDIA GPU (CUDA 12.4)
uv pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu124

# Apple Silicon (自动使用MPS)
uv pip install torch==2.9.0 torchvision==0.24.0
```

### 开发环境

```bash
uv pip sync requirements.txt requirements-dev.txt
```

## 模型下载

主要模型会在首次运行时自动下载：
- SigLIP: `google/siglip-base-patch16-224-i18n` (~400MB)
- BLIP: `Salesforce/blip-image-captioning-base` (~990MB)
- PaddleOCR: 中文识别模型 (~200MB)

## 依赖文件结构

```
blueprints/
├── phase1/
│   ├── requirements.txt      # 核心依赖
│   └── requirements-dev.txt  # 开发工具
└── phase_final/
    ├── requirements.txt      # 生产环境
    └── requirements-dev.txt  # 扩展开发工具
```

---

**更新时间**: 2025年11月12日  
**状态**: 正式版本