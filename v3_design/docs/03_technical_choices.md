# 技术选型 - Vibe Photos V3

## 🎯 选型原则

1. **实用性** > 先进性
2. **易用性** > 功能完整性  
3. **社区支持** > 独特功能
4. **CPU友好** > GPU依赖
5. **渐进增强** > 一步到位

## 🤖 AI模型选择

### 图像理解模型对比

| 模型 | 优点 | 缺点 | 使用场景 | 选择 |
|------|------|------|----------|------|
| **CLIP** | 轻量、CPU友好、零样本 | 精度有限 | 基础分类 | ✅ MVP |
| **RTMDet-L** | Apache许可、高精度(52.8% mAP)、社区支持好 | 需要GPU加速 | 精确物体检测 | ✅ Phase 2 |
| **GroundingDINO** | 开放词汇、灵活 | 资源消耗较大 | 开放词汇检测 | ⚠️ Phase 3 |
| **YOLO v8** | 快速、成熟 | AGPL许可限制、精度一般 | 实时检测 | ❌ |
| **DINOv2** | 强特征提取 | 需要微调 | Few-shot | ✅ Phase 3 |
| **SAM** | 精确分割 | 资源消耗大 | 物体分割 | ⚠️ 可选 |

### OCR模型选择

| 模型 | 中文支持 | 速度 | 准确率 | 选择 |
|------|----------|------|---------|------|
| **PaddleOCR** | 优秀 | 快 | 高 | ✅ 首选 |
| **EasyOCR** | 良好 | 中 | 中 | 备选 |
| **Tesseract** | 一般 | 慢 | 低 | ❌ |

### 推荐组合

```python
# Phase 1 - MVP (CPU友好)
models_mvp = {
    'classifier': 'openai/clip-vit-base-patch32',  # 400MB
    'ocr': None,  # 暂不启用
    'device': 'cpu'
}

# Phase 2 - 增强 (GPU可选)
models_enhanced = {
    'classifier': 'openai/clip-vit-large-patch14',  # 900MB
    'detector': 'rtmdet-l',  # MMDetection, Apache-2.0, 高精度
    'ocr': 'paddleocr-v4',  # 300MB
    'device': 'cuda:0 if available else cpu'
}

# Phase 3 - 专业 (GPU推荐)
models_pro = {
    'classifier': 'clip-vit-large',
    'detector': 'rtmdet-x',  # 或 groundingdino用于开放词汇
    'ocr': 'paddleocr-v4',
    'few_shot': 'dinov2-base',
    'device': 'cuda:0'
}
```

## 💾 数据存储方案

### 数据库选择

| 方案 | 优点 | 缺点 | 适用阶段 |
|------|------|------|----------|
| **SQLite** | 零配置、轻量 | 并发限制 | ✅ MVP |
| **PostgreSQL** | 功能全、pgvector | 需要安装 | ✅ 生产 |
| **MongoDB** | 灵活schema | 资源占用 | ❌ |

### 向量数据库

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| **Faiss** | 快速、成熟 | 仅向量 | ✅ 首选 |
| **Qdrant** | 功能全 | 独立服务 | Phase 3 |
| **Pinecone** | 云服务 | 付费、网络依赖 | ❌ |

### 存储架构

```python
# 混合存储策略
storage_architecture = {
    'metadata': 'SQLite/PostgreSQL',  # 元数据
    'vectors': 'Faiss',  # 向量索引
    'images': 'File System',  # 原始文件
    'thumbnails': 'Cache Directory',  # 缩略图
    'models': 'Local Cache'  # 模型文件
}
```

## 🌐 Web框架选择

### API框架

| 框架 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| **FastAPI** | 异步、自动文档、类型安全 | - | ✅ 首选 |
| **Flask** | 简单、轻量 | 同步 | 备选 |
| **Django** | 功能全 | 过重 | ❌ |

### UI方案

| 方案 | 优点 | 缺点 | 使用场景 |
|------|------|------|----------|
| **Gradio** | 快速原型、零前端 | 定制受限 | ✅ MVP |
| **Streamlit** | 简单、美观 | 性能一般 | 备选 |
| **React** | 灵活、生态好 | 需要前端知识 | ✅ Phase 2 |
| **Vue** | 渐进式 | 生态较小 | 备选 |

## 📦 依赖管理

### Python包管理

```toml
# 使用 uv (用户偏好)
[project]
name = "vibe-photos-v3"
version = "3.0.0"
requires-python = ">=3.11"

dependencies = [
    # 核心 (2024年11月最新版本)
    "torch==2.9.0",              # 最新稳定版
    "transformers==4.57.1",      # 最新稳定版
    "pillow==12.0.0",            # 最新稳定版
    
    # Web
    "fastapi==0.121.1",          # 最新稳定版
    "uvicorn[standard]==0.38.0", # 最新稳定版
    
    # 数据
    "sqlalchemy==2.0.44",        # 最新稳定版  
    "pydantic==2.12.4",          # 最新稳定版
    
    # AI模型
    "clip-interrogator==0.6.0",
    "paddlepaddle==3.2.0",       # 最新稳定版
    "paddleocr==3.3.1",          # 最新稳定版
    
    # 工具
    "typer==0.20.0",             # 最新稳定版
    "rich==14.2.0",              # 最新稳定版
]

[project.optional-dependencies]
gpu = ["torch==2.9.0+cu124"]    # CUDA 12.4支持
dev = ["pytest==9.0.0", "black==25.11.0", "ruff==0.14.4"]  # 最新版本
```

## 🏗 系统架构

### 微服务 vs 单体

**选择：渐进式单体**

```python
# 开始时单体，按需拆分
architecture = {
    'phase1': 'Monolith',  # 简单快速
    'phase2': 'Modular Monolith',  # 模块化
    'phase3': 'Service-Oriented',  # 按需拆分
}
```

### 部署方案

| 环境 | 方案 | 工具 |
|------|------|------|
| 开发 | 本地 | `uv run` |
| 测试 | Docker | `docker-compose` |
| 生产 | 容器/裸机 | `systemd` / `k8s` |

## ⚡ 性能优化技术

### 1. 缓存策略

```python
cache_layers = {
    'L1': 'Memory (LRU)',  # 热数据
    'L2': 'Redis',  # 会话数据
    'L3': 'Disk',  # 持久缓存
}
```

### 2. 异步处理

```python
async_strategies = {
    'web': 'FastAPI async/await',
    'tasks': 'asyncio + ThreadPool',
    'queue': 'Python Queue (简单) / Celery (复杂)'
}
```

### 3. 批处理优化

```python
batch_config = {
    'inference': 16,  # GPU批大小
    'database': 1000,  # 批量插入
    'thumbnail': 100,  # 并行生成
}
```

## 🔐 安全考虑

### 数据安全
- 本地存储（无云端风险）
- 敏感信息脱敏
- 用户数据隔离

### API安全
- JWT认证（如需要）
- Rate limiting
- CORS配置

## 📊 技术栈总结

### MVP技术栈（立即可用）

```yaml
ai:
  model: CLIP-base
  framework: transformers
  device: CPU

storage:
  database: SQLite
  files: Local FS

api:
  framework: FastAPI
  ui: Gradio/CLI

tools:
  package: uv
  cli: typer
  logging: loguru
```

### 生产技术栈（目标）

```yaml
ai:
  models: [CLIP, GroundingDINO, PaddleOCR, DINOv2]
  framework: transformers + custom
  device: CUDA/CPU

storage:
  database: PostgreSQL
  vectors: Faiss
  cache: Redis
  files: Local/S3

api:
  framework: FastAPI
  ui: React
  auth: JWT

monitoring:
  metrics: Prometheus
  logs: Loki
  traces: Jaeger
```

## 🎯 决策矩阵

| 技术领域 | MVP选择 | 生产选择 | 理由 |
|----------|---------|----------|------|
| AI模型 | CLIP | CLIP+GroundingDINO | 平衡准确性和性能 |
| 数据库 | SQLite | PostgreSQL | 从简单到强大 |
| 向量库 | Numpy | Faiss | 规模化需要 |
| Web框架 | FastAPI | FastAPI | 一致性 |
| UI | Gradio | React | 用户体验提升 |
| 部署 | Local | Docker | 标准化 |

## ✅ 最终建议

### 技术选型三原则

1. **先简单，后复杂** - SQLite → PostgreSQL
2. **先单机，后分布** - 单体 → 微服务
3. **先CPU，后GPU** - 优化算法 → 硬件加速

### 关键技术决策

- ✅ **使用CLIP作为基础模型** - 平衡效果和资源
- ✅ **RTMDet-L用于物体检测** - Apache许可、高精度、无法律风险
- ✅ **SQLite起步，PostgreSQL生产** - 渐进式升级
- ✅ **FastAPI贯穿始终** - 现代、高效、一致
- ✅ **Gradio快速原型，React长期** - 快速迭代
- ✅ **本地优先，云端可选** - 数据安全

---

下一步：查看具体实施指南 → [实施指南](04_implementation_guide.md)
