# 技术选型 - Vibe Photos Phase Final

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
| **SigLIP-base-i18n** | 多语言支持、语义理解强、CPU友好 | 需要额外内存 | 基础分类+语义搜索 | ✅ Phase 1/Phase 2 |
| **BLIP-base** | 生成图像描述、物体识别 | 仅英文 | 内容理解+物体检测 | ✅ Phase 1/Phase 2 |
| **SigLIP-large** | 最强语义理解、i18n支持 | 资源消耗大(1.5GB) | 生产环境增强 | ⚠️ Phase 3 |
| **GroundingDINO** | 开放词汇、灵活 | 资源消耗较大 | 开放词汇检测 | ⚠️ Phase 3 |
| **YOLO v8** | 快速、成熟 | AGPL许可限制、精度一般 | 实时检测 | ❌ |
| **RTMDet** | Apache许可、高精度 | mmcv依赖问题、维护困难 | 物体检测 | ❌ 已废弃 |
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
# Phase 1 - 基础验证 (2周)
models_phase1 = {
    'classifier': 'google/siglip-base-patch16-224-i18n',  # 多语言分类
    'captioning': 'Salesforce/blip-image-captioning-base',  # 图像理解
    'ocr': 'paddleocr-v4',  # 中文支持好
    'database': 'sqlite',   # 零配置
    'search': 'fts5',      # SQLite全文搜索
    'device': 'cuda:0 if available else cpu'
}

# Phase 2 - 语义增强 (1个月)
models_phase2 = {
    'classifier': 'google/siglip-base-patch16-224-i18n',  # 保持分类能力
    'captioning': 'Salesforce/blip-image-captioning-base',  # 图像理解
    'detector': 'grounding-dino',  # 开放词汇检测（增强功能）
    'embedder': 'google/siglip-base-patch16-224',  # 语义嵌入(384MB)
    'caption': 'Salesforce/blip-image-captioning-base',  # 可选
    'ocr': 'paddleocr-v4',
    'database': 'sqlite + numpy',  # 简单向量搜索
    'search': 'hybrid (text + vector)',  # 混合搜索
    'device': 'cuda:0 if available else cpu'
}

# Phase 3 - 生产级 (3个月)
models_production = {
    'classifier': 'google/siglip-large-patch16-384-i18n',  # 更强大的多语言支持(1.5GB)
    'captioning': 'Salesforce/blip-image-captioning-large',  # 高质量描述
    'detector': 'grounding-dino',  # 开放词汇检测
    'ocr': 'paddleocr-v4',
    'few_shot': 'dinov2-base',  # Few-shot学习
    'database': 'postgresql + pgvector',  # 统一数据库和向量搜索
    'search': 'advanced hybrid with RRF',  # 高级混合搜索
    'cache': 'redis',  # 性能优化
    'device': 'cuda:0'  # 建议GPU
}
```

## 💾 数据存储方案

### 数据库选择（渐进式升级）

| 方案 | 优点 | 缺点 | 适用阶段 |
|------|------|------|----------|
| **SQLite** | 零配置、轻量 | 并发限制 | ✅ Phase 1 |
| **SQLite + JSON** | 支持向量存储 | 搜索性能受限 | ✅ Phase 2 |
| **PostgreSQL + pgvector** | 原生向量支持、高性能 | 需要安装配置 | ✅ 生产 |
| **MongoDB** | 灵活schema | 资源占用 | ❌ |

### 向量搜索方案（按复杂度递进）

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| **Numpy** | 无依赖、简单 | 仅内存、慢 | ✅ Phase 2 |
| **pgvector** | 数据库集成、统一管理 | 需PostgreSQL | ✅ Phase 3/生产 |
| **Faiss** | 快速、成熟 | 额外维护成本 | ⚠️ 备选 |
| **Qdrant** | 功能全面 | 独立服务 | ❌ 过度复杂 |
| **Pinecone** | 云服务 | 付费、网络依赖 | ❌ |

### 存储架构

```python
# 渐进式存储策略
storage_phase1 = {
    'metadata': 'SQLite',  # 元数据
    'images': 'File System',  # 原始文件
    'thumbnails': 'Cache Directory',  # 缩略图
}

storage_phase2 = {
    'metadata': 'SQLite',  # 元数据
    'vectors': 'JSON in SQLite',  # 向量存储（简单方案）
    'images': 'File System',  # 原始文件
    'thumbnails': 'Cache Directory',  # 缩略图
}

storage_production = {
    'database': 'PostgreSQL + pgvector',  # 统一存储（元数据+向量）
    'cache': 'Redis',  # 缓存层
    'images': 'File System / S3',  # 原始文件
    'thumbnails': 'CDN / Cache',  # 缩略图
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
name = "vibe-photos-phase_final"
version = "3.0.0"
requires-python = ">=3.11"

dependencies = [
    # 核心 (2024年11月最新版本)
    "torch==2.9.0",              # 最新稳定版
    "transformers==4.57.1",      # 最新稳定版
    "pillow==11.3.0",            # 最新稳定版
    
    # Web
    "fastapi==0.121.1",          # 最新稳定版
    "uvicorn[standard]==0.38.0", # 最新稳定版
    
    # 数据
    "sqlalchemy==2.0.44",        # 最新稳定版  
    "pydantic==2.11.10",         # 最新稳定版
    
    # AI模型
    "transformers==4.57.1",      # SigLIP + BLIP
    "sentence-transformers==5.1.2",  # 语义搜索
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
  model: google/siglip-base-patch16-224-i18n
  captioning: Salesforce/blip-image-captioning-base
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
  models: [SigLIP, BLIP, GroundingDINO, PaddleOCR, DINOv2]
  framework: transformers + custom
  device: CUDA/CPU

storage:
  database: PostgreSQL + pgvector  # 统一存储方案
  vectors: pgvector  # 原生向量支持，足够百万级
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

| 技术领域 | Phase 1选择 | Phase 2选择 | 生产选择 | 理由 |
|----------|----------|----------|----------|------|
| 图像分类 | SigLIP-base-i18n | SigLIP-base-i18n | SigLIP-large-i18n | 精度递进 |
| 物体检测 | BLIP-base | BLIP + GroundingDINO | GroundingDINO | 能力增强 |
| 语义理解 | - | SigLIP-base | SigLIP-large | 能力增强 |
| 数据库 | SQLite | SQLite+JSON | PostgreSQL | 从简单到强大 |
| 向量存储 | - | Numpy/JSON | pgvector | 统一管理 |
| Web框架 | FastAPI | FastAPI | FastAPI | 一致性 |
| UI | Streamlit | Gradio | React | 用户体验提升 |
| 部署 | Local | Local | Docker/K8s | 标准化 |

## ✅ 最终建议

### 技术选型三原则

1. **先简单，后复杂** - SQLite → PostgreSQL
2. **先单机，后分布** - 单体 → 微服务
3. **先CPU，后GPU** - 优化算法 → 硬件加速

### 关键技术决策

- ✅ **SigLIP+BLIP组合** - 多语言支持+图像理解，无依赖问题
- ✅ **GroundingDINO增强** - Phase 2开始作为可选增强功能
- ✅ **SQLite起步，PostgreSQL生产** - 渐进式升级
- ✅ **PostgreSQL + pgvector统一存储** - Phase 3避免多系统复杂度
- ✅ **FastAPI贯穿始终** - 现代、高效、一致
- ✅ **UI渐进升级** - Streamlit→Gradio→React
- ✅ **本地优先，云端可选** - 数据安全

---

下一步：查看具体实施指南 → [实施指南](04_implementation_guide.md)
