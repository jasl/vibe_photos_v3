# 最终技术决策文档

> ⚠️ **本文档为最终决策**，所有开发实施请以此为准。调研过程和备选方案请参见[技术调研归档](./research/TECHNOLOGY_RESEARCH_ARCHIVE.md)。

## 📌 核心技术栈

### 🗄 数据存储
| 组件 | 决策 | 理由 |
|------|------|------|
| **数据库** | PostgreSQL 14+ | 成熟、可靠、功能全面 |
| **向量存储** | pgvector扩展 | 统一存储、事务安全、足够3万张照片 |
| **缓存** | Redis | 高性能、支持多种数据结构 |
| **文件存储** | 本地文件系统 → S3 | 渐进式升级 |

### 🤖 AI模型
| 功能 | 模型 | 理由 |
|------|------|------|
| **通用分类** | CLIP-ViT-B/32 | 平衡性能和速度 |
| **物体检测** | RTMDet-L | Apache许可、高精度 |
| **OCR** | PaddleOCR v4 | 中文支持最佳 |
| **Few-shot** | DINOv2-base | 自监督学习效果好 |

### 🔧 应用框架
| 层次 | 技术 | 理由 |
|------|------|------|
| **API框架** | FastAPI | 异步、高性能、自动文档 |
| **任务队列** | Celery + Redis | 成熟、功能完整 |
| **CLI** | Typer + Rich | 现代、美观 |
| **Web UI** | Gradio → React | 渐进式升级 |

### 📊 监控运维
| 功能 | 技术 | 理由 |
|------|------|------|
| **指标监控** | Prometheus + Grafana | 开源标准 |
| **任务监控** | Flower | Celery原生支持 |
| **日志** | Python logging + Rich | 简单够用 |
| **部署** | Docker + Docker Compose | 标准化部署 |

## 🎯 关键决策详解

### 1. 为什么选择 PostgreSQL + pgvector？

**决策**：使用PostgreSQL + pgvector作为统一的数据和向量存储方案

**理由**：
- ✅ **规模适配**：3万张照片，pgvector查询<20ms
- ✅ **架构简单**：单一数据源，无同步问题
- ✅ **事务安全**：ACID保证数据一致性
- ✅ **运维简单**：使用PostgreSQL原生工具
- ✅ **成本效益**：不需要额外的向量数据库

**性能参考**：
```yaml
30K向量:
  索引构建: < 1分钟
  查询延迟: < 20ms
  内存占用: < 500MB
```

### 2. 为什么不用 Faiss？

**Faiss仅作为未来可选方案**，触发条件：
- 向量超过100万
- 查询延迟>200ms
- 需要GPU加速

当前规模（3万）使用Faiss是过度设计。

### 3. 为什么选择 RTMDet 而非 YOLO？

**决策**：使用RTMDet-L替代YOLO系列

**理由**：
- ✅ **许可证**：Apache 2.0（商用友好）vs AGPL（限制多）
- ✅ **性能**：52.8% mAP，满足需求
- ✅ **生态**：OpenMMLab支持完善

### 4. 视频处理策略

**决策**：MVP阶段排除视频处理

**理由**：
- 视频仅占2.5%（762个文件）
- 降低MVP复杂度
- 集中资源优化图片处理（97.5%）

**未来**：Phase 2根据用户需求决定是否支持

## 📦 依赖清单

### 核心依赖
```txt
# API框架
fastapi==0.121.1
uvicorn==0.35.0
pydantic==2.10.4

# 数据库
psycopg2-binary==2.9.9
pgvector==0.2.5
sqlalchemy==2.1.0

# AI模型
torch==2.5.1
transformers==4.47.1
paddleocr==3.3.1

# 任务队列
celery[redis]==5.6.0
redis==6.1.0
flower==2.3.0

# 工具
typer[all]==0.20.0
rich==14.2.0
```

### 可选依赖
```txt
# 仅当向量>100万时
# faiss-cpu==1.12.0

# 视频处理（Phase 2）
# ffmpeg-python==0.3.0
# opencv-python==5.1.0
```

## 🚀 实施路线

### 立即实施
1. 安装PostgreSQL 14+ 和 pgvector
2. 配置Celery + Redis
3. 实现基础CLIP分类
4. 构建FastAPI接口

### 短期计划（2周）
1. 集成RTMDet物体检测
2. 添加PaddleOCR
3. 实现批量处理
4. 优化查询性能

### 中期计划（1月）
1. Few-shot学习
2. Web UI开发
3. 性能优化
4. 部署脚本

## ⚠️ 重要提醒

1. **不要过度设计**：当前规模不需要Faiss、Elasticsearch等
2. **保持简单**：优先使用PostgreSQL生态内的方案
3. **渐进升级**：从SQLite开始，需要时再迁移到PostgreSQL
4. **延迟决策**：视频等功能等用户反馈后再决定

## 📋 检查清单

开发前确认：
- [ ] PostgreSQL 14+已安装
- [ ] pgvector扩展已启用
- [ ] Redis已运行
- [ ] Python 3.11+环境
- [ ] CUDA可用（可选）

## 🔗 相关文档

- [系统架构](./architecture/system_architecture.md)
- [实施指南](./docs/04_implementation_guide.md)
- [技术调研归档](./research/TECHNOLOGY_RESEARCH_ARCHIVE.md)

---

**最后更新**: 2024年11月
**状态**: 最终决策 ✅
**下次评审**: 当向量规模接近100万时
