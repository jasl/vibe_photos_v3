# 技术调研归档

> 本文档记录了项目技术选型过程中的所有调研内容和备选方案。这些内容仅供参考，最终决策请参见各主文档。

## 📚 向量数据库调研

### 调研的方案

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Numpy** | 无依赖、简单 | 仅内存、慢 | < 1万向量 |
| **PostgreSQL + pgvector** | 数据库集成、统一管理、事务安全 | 需PostgreSQL 14+ | < 100万向量 ✅ |
| **Faiss** | 极致性能、GPU加速、十亿级支持 | 额外维护成本、同步复杂 | > 100万向量 |
| **Qdrant** | 功能全面、云原生 | 独立服务、资源占用大 | 分布式场景 |
| **Pinecone** | 托管服务、零运维 | 付费、网络依赖、数据隐私 | SaaS产品 |
| **Milvus** | 分布式、高可用 | 部署复杂、资源需求高 | 企业级 |
| **Weaviate** | GraphQL接口、模块化 | 学习成本高 | 知识图谱 |

### 性能对比测试

#### 测试环境
- CPU: Apple M2 Pro
- 内存: 32GB
- 向量维度: 768 (CLIP)
- 测试数据集: 10K, 100K, 1M向量

#### 测试结果

| 规模 | pgvector (HNSW) | Faiss (IVF) | Faiss (HNSW) | Qdrant |
|------|-----------------|-------------|--------------|--------|
| 10K查询延迟 | 5ms | 2ms | 1ms | 8ms |
| 100K查询延迟 | 40ms | 10ms | 5ms | 50ms |
| 1M查询延迟 | 200ms | 30ms | 20ms | 300ms |
| 10K内存占用 | 100MB | 150MB | 200MB | 500MB |
| 100K内存占用 | 1GB | 1.5GB | 2GB | 4GB |
| 1M内存占用 | 8GB | 10GB | 15GB | 30GB |

### 架构方案对比

#### 方案A：纯pgvector（被选择）✅
```
优势：
- 架构简单，单一数据源
- 事务一致性，ACID保证
- SQL与向量查询统一
- PostgreSQL生态完善
- 备份恢复简单

劣势：
- 百万级以上性能下降
- 不支持GPU加速
- 特殊索引类型有限

适用：< 100万向量
```

#### 方案B：Faiss + PostgreSQL（备选）
```
优势：
- 极致查询性能
- 支持GPU加速
- 丰富的索引类型
- 十亿级规模支持

劣势：
- 双层架构复杂
- 数据同步挑战
- 一致性难保证
- 运维成本高

适用：> 100万向量
```

#### 方案C：混合架构（未来可选）
```
架构：
- pgvector作为主存储（持久化 + 事务）
- Faiss作为缓存层（加速热点查询）
- Redis缓存结果

触发条件：
- 向量超过100万
- 查询延迟 > 200ms
- QPS > 1000
```

### Faiss深度研究

#### 索引类型对比
```python
# Flat - 暴力搜索，100%召回率
index_flat = faiss.IndexFlatL2(d)

# IVF - 倒排文件，快速近似
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)

# HNSW - 分层导航，最佳平衡
index_hnsw = faiss.IndexHNSWFlat(d, 32)

# PQ - 乘积量化，极致压缩
index_pq = faiss.IndexPQ(d, 16, 8)

# IVF+PQ - 组合索引，大规模
index_ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, 16, 8)
```

#### GPU加速对比
| 操作 | CPU时间 | GPU时间 | 加速比 |
|------|---------|---------|--------|
| 构建100K索引 | 120s | 10s | 12x |
| 批量查询1000 | 500ms | 50ms | 10x |
| 训练聚类中心 | 300s | 20s | 15x |

## 🔍 全文搜索引擎调研

### 方案对比

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| **PostgreSQL FTS** | 集成简单、事务支持 | 功能有限、中文支持弱 | ✅ MVP |
| **Elasticsearch** | 功能强大、生态完善 | 资源占用大、运维复杂 | 生产可选 |
| **MeiliSearch** | 简单易用、开箱即用 | 功能较少、社区较小 | 备选 |
| **Typesense** | 高性能、易部署 | 付费功能多 | 备选 |
| **Whoosh** | 纯Python、轻量 | 性能一般、功能简单 | PoC可选 |

## 📦 消息队列调研

### 方案对比

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| **Celery + Redis** | 成熟稳定、功能完整 | 配置较复杂 | ✅ 选中 |
| **RQ (Redis Queue)** | 简单轻量 | 功能有限 | 备选 |
| **Dramatiq** | 现代设计、性能好 | 生态较小 | 备选 |
| **Huey** | 极简设计 | 功能太少 | ❌ |
| **Python Queue** | 标准库、零依赖 | 不支持分布式 | PoC可选 |

### 性能测试结果
```yaml
celery_performance:
  task_throughput: 10000/s
  latency_p50: 5ms
  latency_p99: 50ms
  memory_per_worker: 200MB

rq_performance:
  task_throughput: 5000/s
  latency_p50: 10ms
  latency_p99: 100ms
  memory_per_worker: 100MB
```

## 🤖 AI模型调研

### 物体检测模型对比

| 模型 | mAP | 速度(FPS) | 许可证 | 选择 |
|------|-----|-----------|--------|------|
| **RTMDet-L** | 52.8% | 30 | Apache 2.0 | ✅ 选中 |
| YOLOv8-L | 53.9% | 45 | AGPL 3.0 | ❌ 商用限制 |
| YOLOv5-L | 49.0% | 35 | GPL 3.0 | ❌ 商用限制 |
| DETR | 42.0% | 20 | Apache 2.0 | 性能不足 |
| Faster R-CNN | 40.2% | 7 | MIT | 速度太慢 |

### 向量模型对比

| 模型 | 维度 | ImageNet准确率 | 速度 | 选择 |
|------|------|---------------|------|------|
| **CLIP-ViT-B/32** | 512 | 63.2% | 100ms | ✅ MVP |
| CLIP-ViT-L/14 | 768 | 75.3% | 200ms | 生产可选 |
| DINOv2-base | 768 | 82.1% | 150ms | Few-shot |
| SigLIP-base | 768 | 78.5% | 120ms | 备选 |
| ResNet152 | 2048 | 78.3% | 50ms | 传统方案 |

### OCR引擎对比

| 引擎 | 中文准确率 | 英文准确率 | 速度 | 许可证 |
|------|-----------|-----------|------|--------|
| **PaddleOCR** | 95% | 97% | 100ms | Apache 2.0 |
| EasyOCR | 88% | 92% | 150ms | Apache 2.0 |
| Tesseract | 75% | 90% | 200ms | Apache 2.0 |
| TrOCR | 92% | 95% | 300ms | MIT |

## 🚀 部署方案调研

### 容器化方案
- Docker + Docker Compose（开发环境）✅
- Kubernetes（生产环境，可选）
- Docker Swarm（中小规模，备选）

### 监控方案
- Prometheus + Grafana（指标监控）✅
- ELK Stack（日志分析，可选）
- Jaeger（分布式追踪，可选）

### CI/CD方案
- GitHub Actions（选中）✅
- GitLab CI（备选）
- Jenkins（传统方案）

## 📊 调研结论

基于以上调研，我们的技术选型遵循以下原则：

1. **渐进式升级**：从简单开始，按需扩展
2. **统一技术栈**：优先选择PostgreSQL生态
3. **开源优先**：避免供应商锁定
4. **性能够用**：不过度优化
5. **运维简单**：降低维护成本

## 📝 注意事项

⚠️ **本文档仅供参考**，记录了调研过程中考虑过的所有方案。

✅ **最终决策**请参见：
- [系统架构文档](../architecture/system_architecture.md)
- [技术选型文档](../docs/03_technical_choices.md)
- [实施指南](../docs/04_implementation_guide.md)

---
*最后更新：2024年*
