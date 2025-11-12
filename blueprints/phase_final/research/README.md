# 研究与归档文档

本目录包含项目开发过程中的研究资料、中间文档和历史记录，仅供参考。

## 📁 文档说明

### 技术研究
- **TECHNOLOGY_RESEARCH_ARCHIVE.md** - 技术选型调研记录
  - 向量数据库对比（Faiss vs Qdrant vs pgvector）
  - 模型选择研究（SigLIP vs CLIP vs RTMDet）
  - 性能测试数据

### 历史记录
- **REVIEW_REPORT_ARCHIVE.md** - 方案审查报告归档
  - 专家反馈记录
  - 初始技术选型（已过时）
  - RTMDet集成方案（已废弃）

- **OPTIMIZATION_SUMMARY.md** - 优化过程总结
  - 架构简化记录
  - 性能优化历程

- **DOCUMENTATION_CLEANUP_SUMMARY.md** - 文档整理记录
  - 文档重组过程
  - 术语统一记录

## ⚠️ 重要说明

1. **这些是历史文档**，可能包含已过时的信息
2. **最新决策**请参考：
   - [../FINAL_TECHNOLOGY_DECISIONS.md](../FINAL_TECHNOLOGY_DECISIONS.md)
   - [../docs/03_technical_choices.md](../docs/03_technical_choices.md)
3. **当前技术栈**：
   - 图像理解：SigLIP + BLIP（替代了RTMDet）
   - 向量存储：PostgreSQL + pgvector（统一方案）
   - 任务队列：Celery + Redis

## 🔄 技术演进历程

```
初始方案（已废弃）
├── RTMDet - 物体检测
│   问题：mmcv依赖无法在Python 3.11+安装
└── CLIP - 图像分类
    问题：仅支持英文，功能受限

最终方案（当前使用）
├── SigLIP - 多语言图像分类
│   优势：18+语言支持，零样本学习
└── BLIP - 图像理解
    优势：生成自然语言描述
```

## 📝 查阅建议

如需了解技术决策背景，建议按以下顺序查阅：
1. 查看最新的技术决策文档
2. 如需了解历史，再查阅本目录文档
3. 注意区分"已采用"和"已废弃"的方案
