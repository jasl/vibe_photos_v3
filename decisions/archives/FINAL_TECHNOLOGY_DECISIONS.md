# 技术决策文档

## 📌 核心技术栈

### Phase 1 - 已确定

| 组件 | 技术选型 | 版本 |
|------|----------|------|
| **语言** | Python | 3.12 |
| **包管理** | uv | 最新 |
| **API框架** | FastAPI | 0.121.1 |
| **UI框架** | Streamlit | 1.51.0 |
| **数据库** | SQLite | 内置 |
| **图像识别** | SigLIP + BLIP | 4.57.1 |
| **OCR** | PaddleOCR | 3.3.1 |
| **深度学习** | PyTorch | 2.9.1 |

### Phase Final - 规划中

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| **数据库** | PostgreSQL + pgvector | 大规模部署时 |
| **缓存** | Redis | 高并发场景 |
| **任务队列** | Celery | 异步处理 |
| **监控** | Prometheus + Grafana | 生产环境 |

## 🎯 关键决策

### 1. SigLIP + BLIP 组合
**原因**：
- 多语言支持（中英日）
- 零样本学习能力
- 无依赖冲突
- Hugging Face生态支持

### 2. PaddleOCR
**原因**：
- 中文识别效果最佳
- 支持多种语言
- 模型轻量高效

### 3. SQLite → PostgreSQL 渐进式升级
**原因**：
- 开发阶段简单高效
- 生产环境平滑迁移
- 向量搜索能力（pgvector）

## 📦 模型选择

| 功能 | 模型 | 大小 |
|------|------|------|
| **图像理解** | google/siglip-base-patch16-224-i18n | ~400MB |
| **图像描述** | Salesforce/blip-image-captioning-base | ~990MB |
| **OCR检测** | PaddleOCR PP-OCRv4 | ~200MB |

---

**状态**: 正式版本  
**更新**: 2025年11月