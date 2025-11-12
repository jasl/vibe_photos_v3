# 依赖管理报告

## 📅 更新日期：2025年11月12日

## 📊 当前依赖状态

### Phase 1 依赖（验证阶段）

#### 核心依赖
| 类别 | 依赖包 | 版本 | 用途 |
|------|--------|------|------|
| **Web框架** | fastapi | 0.121.1 | API服务 |
| | uvicorn | 0.38.0 | ASGI服务器 |
| | streamlit | 1.51.0 | 快速UI原型 |
| **数据处理** | sqlalchemy | 2.0.44 | 数据库ORM |
| | pydantic | 2.11.10 | 数据验证 |
| | pyyaml | 6.0.2 | 配置文件 |
| **AI模型** | torch | 2.9.0 | 深度学习框架 |
| | torchvision | 0.24.0 | 计算机视觉 |
| | transformers | 4.57.1 | SigLIP/BLIP模型 |
| | sentence-transformers | 5.1.2 | 语义搜索 |
| **OCR** | paddlepaddle | 3.2.0 | OCR框架 |
| | paddleocr | 3.3.1 | 文字识别 |
| **工具库** | pillow | 11.3.0 | 图像处理 |
| | numpy | 2.3.4 | 数值计算 |
| | requests | 2.32.3 | HTTP请求 |
| | aiofiles | 24.1.0 | 异步文件操作 |
| | python-multipart | 0.0.20 | 文件上传 |

**总计**: 17个核心依赖

### Phase Final 依赖（生产阶段）

在Phase 1基础上，增加以下可选模块：

#### 可选功能
| 功能 | 依赖包 | 版本 | 说明 |
|------|--------|------|------|
| **向量搜索** | psycopg2-binary | 2.9.9 | 需要时启用 |
| | pgvector | 0.2.5 | PostgreSQL向量扩展 |
| **备选UI** | gradio | 5.49.1 | 可替代Streamlit |
| **监控** | loguru | 0.7.3 | 日志管理 |
| | prometheus-client | 0.23.1 | 性能监控 |

## ✅ 核心功能确认

### 必需功能（Phase 1即包含）
- ✅ **SigLIP + BLIP**: 图像理解和描述生成
- ✅ **PaddleOCR**: 中文文字识别
- ✅ **语义搜索**: sentence-transformers支持
- ✅ **Web服务**: FastAPI + Uvicorn
- ✅ **用户界面**: Streamlit

### 可选扩展（Phase Final）
- ⭕ 向量数据库（大规模部署时）
- ⭕ 高级监控（生产环境）
- ⭕ 备选UI方案

## 🚀 快速开始

### 基础安装
```bash
# Phase 1 - 包含所有核心功能
cd blueprints/phase1
pip install -r requirements.txt
```

### 开发环境
```bash
# 添加开发工具
pip install -r requirements-dev.txt
```

## 📝 版本说明

所有依赖均为2024年11月最新稳定版本，经过兼容性验证。

---

**状态**: ✅ 正式版本  
**维护者**: Vibe Photos Team