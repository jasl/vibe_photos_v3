# 🚀 Vibe Photos 产品路线图

## 📋 执行摘要

基于Gemini Deep Think的反馈和现有设计评估，我们制定了一个**渐进式三阶段路线图**，在保持实用性的同时逐步引入先进技术。

### 核心策略
- **Phase 1**: 验证核心功能，建立基础架构
- **Phase 2**: 增强语义理解，引入智能搜索
- **PoC3/Phase 3**: 生产级优化，完整功能集

### ⚠️ POC阶段特别说明
- **无兼容性约束**: POC阶段可随时进行破坏性改动
- **无数据迁移**: 每个版本可以重新开始，不考虑历史数据
- **快速迭代**: 专注于功能验证，不追求稳定性
- **缓存可选**: cache目录可随时清空重建

### 🔧 Python环境管理（统一使用uv）
**本项目必须使用 `uv` 管理所有Python依赖**
```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或 brew install uv (macOS)
```
- ✅ 使用 `uv venv` 创建虚拟环境
- ✅ 使用 `uv add/remove` 管理依赖
- ✅ 使用 `uv run` 执行脚本
- ❌ 禁止使用 pip/pip-tools/poetry/conda

## 🎯 阶段规划

### Phase 1: 基础功能验证（2周）
**目标**: 验证物体检测和搜索的可行性

#### 技术栈
```yaml
核心技术:
  检测器: RTMDet-L (Apache-2.0, 52.8% mAP)
  OCR: PaddleOCR v4
  数据库: SQLite + FTS5
  API: FastAPI
  UI: Streamlit

主要功能:
  - 测试数据集处理（samples目录）
  - 增量处理支持
  - 图像预处理（归一化、缩略图、去重）
  - 物体检测（80类COCO）
  - 文本提取（中英文）
  - 全文搜索
  - 简单Web界面（无上传功能）

预留设计:
  - 向量存储字段（embedding_json）
  - 混合搜索接口（SearchEngine类）
  - 模块化架构
```

#### 缓存复用优势
- **10倍性能提升**: 从10张/分钟提升到100+张/分钟
- **跨版本共享**: Phase 2、PoC3可直接复用
- **增量处理**: 只处理新增图片
- **存储优化**: 相同内容共享缓存（基于感知哈希）

#### 成功指标
- [ ] 处理1000张图片无崩溃
- [ ] 检测准确率 > 70%
- [ ] 搜索响应 < 1秒
- [ ] 用户可在5分钟内上手
- [ ] 缓存命中率 > 90%（增量处理时）

#### 交付物
- 可运行的原型系统
- 基础API文档
- 性能测试报告
- 用户反馈收集

---

### Phase 2: 语义搜索增强（1个月）
**目标**: 实现智能语义搜索，提升用户体验

#### 新增技术
```yaml
语义理解:
  嵌入模型: SigLIP-base (google/siglip-base-patch16-224)
  图像描述: BLIP-base (可选)
  向量搜索: Numpy + Cosine Similarity
  
搜索增强:
  - 自然语言查询："找到所有iPhone照片"
  - 混合搜索：文本 + 语义
  - 相似图片查找
  - 中英文混合搜索

架构升级:
  - 双模型并行：RTMDet + SigLIP
  - 简单向量索引（JSON存储）
  - 搜索结果融合算法
```

#### Phase 1 图像处理流水线（带缓存）
```python
# 完整的处理流水线
class ImagePipeline:
    """
    目录结构：
    - samples/  : 原始数据（只读）
    - cache/    : 可复用缓存（跨版本共享）
    - data/     : 数据库和状态
    
    处理流程：
    1. 感知哈希计算 → 去重 + 缓存键
    2. 格式归一化 → cache/images/processed/
    3. 缩略图生成 → cache/images/thumbnails/
    4. 物体检测 → cache/detections/
    5. 文本提取 → cache/ocr/
    6. 索引更新 → data/vibe_photos.db
    """
    
    def process(self, dataset_dir='samples'):
        # 扫描只读数据集
        images = scan_dataset(dataset_dir, readonly=True)
        
        # 增量处理（跳过已处理）
        new_images = filter_processed(images)
        
        # 批量处理（带缓存）
        for batch in batches(new_images):
            # 使用感知哈希作为缓存键
            for image in batch:
                phash = compute_phash(image)
                
                # 检查缓存
                if cache_exists(phash):
                    load_from_cache(phash)
                else:
                    # 计算并缓存
                    result = process_image(image)
                    save_to_cache(phash, result)
```

#### 关键实现
```python
class EnhancedProcessor:
    def __init__(self):
        self.detector = RTMDetDetector()      # 物体检测
        self.embedder = SigLIPEmbedder()      # 语义嵌入
        self.ocr = PaddleOCR()                # 文本提取
    
    def process_image(self, image_path):
        # 并行处理
        detection = self.detector.detect(image_path)
        embedding = self.embedder.encode(image_path)
        text = self.ocr.extract(image_path)
        
        return {
            'objects': detection,
            'embedding': embedding.tolist(),  # 存为JSON
            'text': text
        }

class HybridSearch:
    def search(self, query, mode='hybrid'):
        if mode == 'text':
            return self.text_search(query)
        elif mode == 'vector':
            return self.vector_search(query)
        else:  # hybrid
            text_results = self.text_search(query)
            vector_results = self.vector_search(query)
            return self.merge_results(text_results, vector_results)
```

#### 成功指标
- [ ] 语义搜索准确率 > 80%
- [ ] 支持自然语言查询
- [ ] 混合搜索效果优于纯文本搜索
- [ ] 处理5000张图片稳定运行

#### 交付物
- 增强版搜索系统
- 语义搜索演示
- A/B测试结果
- 性能优化报告

---

### PoC3/Phase 3: 生产级系统（2-3个月）
**目标**: 构建可扩展的生产系统

#### 为什么选择PostgreSQL + pgvector？
1. **统一管理**: 一个数据库同时处理结构化数据和向量数据
2. **原生支持**: pgvector是PostgreSQL的官方扩展，稳定可靠
3. **简化运维**: 无需维护额外的向量数据库服务
4. **成本效益**: 避免Faiss的额外维护成本和复杂性
5. **生产就绪**: PostgreSQL是业界标准，运维经验丰富

#### 完整技术栈
```yaml
高级模型:
  检测器: RTMDet-X (更高精度)
  嵌入: SigLIP-large-i18n (多语言)
  描述: BLIP-large 或 LMM
  学习: DINOv2 (Few-shot)
  
基础设施:
  数据库: PostgreSQL + pgvector（统一存储和向量搜索）
  缓存: Redis
  队列: Celery（可选）
  
搜索能力:
  - 高级混合搜索（RRF算法）
  - 实时索引更新
  - 个性化排序
  - 搜索建议
  
用户功能:
  - Few-shot学习
  - 批量标注工具
  - 导出功能
  - 用户偏好学习
```

#### PostgreSQL + pgvector 实施细节
```sql
-- 安装pgvector扩展
CREATE EXTENSION vector;

-- 创建统一的图片表
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    filepath TEXT NOT NULL,
    
    -- 向量嵌入（SigLIP输出，1024维）
    embedding vector(1024),
    
    -- 元数据和搜索字段
    caption TEXT,
    ocr_text TEXT,
    objects JSONB,
    
    -- 索引
    created_at TIMESTAMP DEFAULT NOW()
);

-- 创建向量索引（HNSW算法，余弦相似度）
CREATE INDEX ON images USING hnsw (embedding vector_cosine_ops);

-- 创建文本搜索索引
CREATE INDEX ON images USING gin(to_tsvector('simple', 
    coalesce(caption, '') || ' ' || coalesce(ocr_text, '')
));
```

#### 架构设计
```
┌─────────────────────────────────────┐
│          Load Balancer              │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│          API Gateway                │
│         (FastAPI)                   │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│       Service Layer                 │
│  ┌─────────┐ ┌─────────┐ ┌────────┐│
│  │Detection│ │Embedding│ │Search  ││
│  │Service  │ │Service  │ │Service ││
│  └─────────┘ └─────────┘ └────────┘│
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│         Data Layer                  │
│  ┌────────────────────┐ ┌─────────┐│
│  │PostgreSQL+pgvector │ │  Redis  ││
│  │(统一存储+向量搜索)│ │ (缓存)  ││
│  └────────────────────┘ └─────────┘│
└─────────────────────────────────────┘
```

#### 成功指标
- [ ] 支持10万+图片
- [ ] 搜索响应 < 500ms (P95)
- [ ] 系统可用性 > 99.5%
- [ ] Few-shot学习准确率 > 85%
- [ ] 用户满意度 > 8/10

#### 交付物
- 生产级系统
- 完整文档
- 部署脚本
- 运维手册
- 性能基准报告

## 📊 技术演进矩阵

| 组件 | Phase 1 | Phase 2 | Phase 3 |
|------|------|------|---------|
| **物体检测** | RTMDet-L | RTMDet-L | RTMDet-X |
| **语义理解** | - | SigLIP-base | SigLIP-large-i18n |
| **图像描述** | - | BLIP-base (可选) | BLIP-large/LMM |
| **OCR** | PaddleOCR | PaddleOCR | PaddleOCR |
| **Few-shot** | - | - | DINOv2 |
| **数据库** | SQLite | SQLite+JSON | PostgreSQL |
| **向量索引** | - | Numpy | pgvector |
| **搜索模式** | 文本 | 文本+向量 | 高级混合(RRF) |
| **缓存** | 文件系统 | 内存+文件 | Redis |
| **部署** | 单机 | 单机 | 分布式可选 |

## 🎯 关键决策点

### Phase 1 → Phase 2 决策门
**评估时间**: Phase 1完成后第1周

评估项目:
- [ ] 物体检测质量是否满足需求？
- [ ] 用户是否需要语义搜索？
- [ ] 纯文本搜索的局限性有多大？
- [ ] 是否有足够资源继续？

**Go/No-Go标准**:
- Go: 检测准确率>70%，用户需要更智能的搜索
- No-Go: 基础功能已满足需求，或资源不足

### Phase 2 → Phase 3 决策门
**评估时间**: Phase 2完成后第2周

评估项目:
- [ ] 语义搜索提升是否明显？
- [ ] 系统规模是否需要升级？
- [ ] 是否需要Few-shot学习？
- [ ] ROI是否合理？

**Go/No-Go标准**:
- Go: 用户量增长，需要更高性能和功能
- No-Go: 当前系统已满足需求

## 📈 风险管理

### 技术风险
| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| SigLIP性能不佳 | Phase 2失败 | 备选CLIP-large，降级到纯文本 |
| 向量搜索太慢 | 用户体验差 | 使用缓存，限制搜索范围 |
| 模型内存过大 | 部署困难 | 模型量化，使用轻量版本 |
| 混合搜索复杂 | 开发延期 | 简化融合算法，渐进实现 |

### 项目风险
| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 需求变更 | 返工 | 敏捷开发，快速迭代 |
| 资源不足 | 延期 | 分阶段交付，优先级管理 |
| 用户接受度低 | 项目失败 | 早期用户参与，持续反馈 |

## 🚦 里程碑时间线

```
2024年11月 - 12月
├── Week 1-2: Phase 1 开发
│   ├── Week 1: 核心功能实现
│   └── Week 2: 测试和优化
│
├── Week 3: Phase 1 评估和决策
│
├── Week 4-7: Phase 2 开发
│   ├── Week 4-5: SigLIP集成
│   ├── Week 6: 混合搜索实现
│   └── Week 7: 测试和优化
│
└── Week 8: Phase 2 评估和规划

2025年1月 - 3月
├── Month 1: 基础设施升级
│   ├── PostgreSQL迁移
│   └── 向量索引优化
│
├── Month 2: 高级功能开发
│   ├── Few-shot学习
│   └── 高级搜索
│
└── Month 3: 生产部署
    ├── 性能优化
    ├── 监控系统
    └── 文档完善
```

## ✅ 行动计划

### 立即行动（本周）
1. [ ] 完成Phase 1环境搭建
2. [ ] 实现RTMDet集成
3. [ ] 建立基础数据库
4. [ ] 创建简单UI

### 下周计划
1. [ ] 完成批处理流程
2. [ ] 实现搜索功能
3. [ ] 进行性能测试
4. [ ] 收集用户反馈

### 长期规划
1. [ ] 建立持续集成
2. [ ] 准备技术文档
3. [ ] 培训运维团队
4. [ ] 制定扩展计划

## 📝 成功要素

1. **保持简单**: 不过度工程化
2. **快速迭代**: 2周一个版本
3. **数据驱动**: 基于指标决策
4. **用户中心**: 持续收集反馈
5. **技术务实**: 选择成熟方案
6. **风险可控**: 渐进式升级
7. **文档完善**: 知识传承

## 🎉 愿景

通过这个渐进式路线图，我们将构建一个既实用又先进的图像搜索系统：

- **短期（1个月）**: 可用的基础系统
- **中期（3个月）**: 智能的语义搜索
- **长期（6个月）**: 生产级AI平台

让我们从Phase 1开始，一步一个脚印地实现这个愿景！

---

*更新时间: 2024年11月*
*基于Gemini Deep Think反馈整合*
