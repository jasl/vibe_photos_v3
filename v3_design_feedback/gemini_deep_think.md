好的，这是一份为你准备的 PoC1 架构重构和 SOTA 技术实施的完整报告。本报告总结了我们讨论的核心结论，推荐了当前最先进（SOTA）的模型选型，并提供了详细的实施细节和代码结构，以便 Coding AI 能够准确、高效地实现这一先进的图像搜索系统。

-----

# PoC1 架构重构与 SOTA 实施指南报告

## I. 执行摘要

本项目旨在解决在海量（3万+张）多样化素材库中快速检索特定“物体”（如“iPhone”或“披萨”）的需求。原始 PoC1 设计文档在工程化方面表现出色，但其核心 AI 技术选型（RTMDet + SQLite FTS）无法满足灵活的语义理解和跨语言搜索需求。

本报告提出了一套重构方案，将核心架构从“传统分类”升级为“现代多模态混合搜索”。该方案引入了最先进的视觉-语言模型（VLM）、向量数据库和多模型融合流水线。

**核心变更总结：**

| 方面 | 原 PoC1 设计 | 修订后的 SOTA 方案 | 核心优势 |
| :--- | :--- | :--- | :--- |
| **基础设施** | SQLite | **PostgreSQL + pgvector** | 支持高效向量搜索和混合搜索 |
| **核心 VLM** | RTMDet | **SigLIP Large (i18n)** | 实现语义理解、零样本识别和原生多语言支持 |
| **图像描述** | (缺失) | **BLIP Base + 翻译模型** | 生成多语言自然语言描述，增强关键词搜索 |
| **文本提取** | PaddleOCR | **PaddleOCR** (保留) | 处理截图和文档的关键信息 |
| **搜索策略** | 仅关键词搜索 (FTS) | **混合搜索** (ANN + FTS + RRF) | 结合语义理解和精确匹配，提高准确率 |

## II. 推荐架构与 SOTA 技术栈

我们建议保持原有的“单体优先”和“批处理”的工程架构（FastAPI + Streamlit），但对其核心组件进行升级。

### 2.1 识别引擎升级（SOTA 组合）

我们推荐一个多模型融合流水线，替代 RTMDet，以提取多维度的图像特征：

| 任务 | 推荐 SOTA 模型 | 模型名称/库 | 作用 |
| :--- | :--- | :--- | :--- |
| **1. 语义嵌入** | SigLIP Large (i18n) | `google/siglip-large-patch16-384-i18n` | ⭐ 核心。生成全局语义向量（1024维），实现跨语言语义搜索。 |
| **2. 图像描述** | BLIP Base | `Salesforce/blip-image-captioning-base` | 生成自然语言描述（英文）。未来可升级为 LMM（如 Qwen-VL）。 |
| **3. 文本翻译** | MarianMT | `Helsinki-NLP/opus-mt-en-zh` | 将 BLIP 描述翻译为中文，优化混合搜索效果。 |
| **4. OCR 提取** | PaddleOCR | `paddleocr` (lang="ch") | 提取截图、文档中的精确文本信息。 |

### 2.2 核心搜索策略：混合搜索 (Hybrid Search)

系统应实现混合搜索，结合语义搜索（ANN）和关键词搜索（FTS），并使用 **RRF (Reciprocal Rank Fusion，倒数排名融合)** 算法合并结果。

## III. 详细实施指南（供 Coding AI 执行）

本节提供了关键的实施细节。

### 3.1 依赖管理 (requirements.txt 更新)

移除 MMDetection 相关依赖，确保包含以下核心库。根据硬件环境选择合适的 `torch` 和 `paddlepaddle` 版本（CPU 或 GPU）。

```text
# 基础框架与数据库 (保持原有版本或使用最新稳定版)
fastapi
uvicorn
streamlit
sqlalchemy>=2.0
pydantic
pillow
psycopg2-binary  # PostgreSQL 驱动
pgvector         # pgvector 的 Python 库支持 (与 SQLAlchemy 集成时需要)

# AI 核心依赖 (PyTorch 生态)
torch
torchvision

# Hugging Face 生态 (用于 SigLIP, BLIP, Translation)
transformers>=4.40.0
accelerate
sentencepiece  # 翻译模型需要

# OCR 引擎 (PaddlePaddle 生态)
# 如果使用GPU，请安装 paddlepaddle-gpu；否则安装 paddlepaddle
paddlepaddle
paddleocr>=2.7
```

### 3.2 数据库迁移与 Schema 设计 (PostgreSQL)

必须正确配置 PostgreSQL。

**关键配置：向量维度**
`google/siglip-large-patch16-384-i18n` 模型的输出维度是 **1024**。

#### 3.2.1 SQL Schema 与索引

```sql
-- 1. 启用 pgvector 扩展 (在数据库中运行一次)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. 图片主表结构 (images)
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL UNIQUE,
    -- (其他元数据字段...)
    process_status TEXT DEFAULT 'pending',

    -- 核心特征存储
    -- SigLIP Large 向量 (维度必须是 1024)
    embedding vector(1024),
    
    -- BLIP + Translation 生成的描述
    caption_en TEXT,
    caption_zh TEXT,
    
    -- PaddleOCR 提取的文本
    ocr_content TEXT
);

-- 3. 创建索引

-- 3.1 向量索引 (用于语义搜索 ANN)
-- 使用 HNSW 算法进行高效的 ANN 搜索。使用余弦距离 (vector_cosine_ops)。
CREATE INDEX ON images USING hnsw (embedding vector_cosine_ops);

-- 3.2 全文搜索索引 (用于关键词搜索 FTS)
-- 关键：分开使用 'english' 和 'simple' 配置，以优化英文内容的索引效果。
CREATE INDEX idx_fts_search ON images USING gin(
    to_tsvector('english', coalesce(caption_en, '')) || 
    to_tsvector('simple', coalesce(caption_zh, '') || ' ' || coalesce(ocr_content, ''))
);
```

#### 3.2.2 SQLAlchemy 模型更新

如果使用 SQLAlchemy ORM，需要在模型定义中引入 `pgvector.sqlalchemy.Vector`。

```python
# app/models.py (关键修改)
from sqlalchemy import Column, Integer, Text
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector # 引入 Vector 类型

Base = declarative_base()

class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    # ... (其他字段)
    
    embedding = Column(Vector(1024)) # 维度 1024
    caption_en = Column(Text)
    caption_zh = Column(Text)
    ocr_content = Column(Text)
```

### 3.3 识别流水线实现 (`processors/engine.py`)

实现一个 `ImageProcessor` 类来管理所有模型。重点关注设备管理、L2 归一化和内存优化。

#### 3.3.1 结构与初始化

```python
import torch
from transformers import AutoProcessor, SiglipModel, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
from PIL import Image
from paddleocr import PaddleOCR
import gc
import logging
from typing import List

# 设备管理函数
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class ImageProcessor:
    def __init__(self):
        self.device = get_device()
        logging.info(f"ImageProcessor initialized on device: {self.device}")
        
        # SOTA Model Configuration
        self.SIGLIP_MODEL = "google/siglip-large-patch16-384-i18n"
        self.BLIP_MODEL = "Salesforce/blip-image-captioning-base"
        self.TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-zh" # EN to ZH
        
        self._load_models()

    def _load_models(self):
        # 1. SigLIP (Multilingual Embedding)
        self.siglip_processor = AutoProcessor.from_pretrained(self.SIGLIP_MODEL)
        self.siglip_model = SiglipModel.from_pretrained(self.SIGLIP_MODEL).to(self.device).eval()

        # 2. BLIP (Captioning)
        self.blip_processor = AutoProcessor.from_pretrained(self.BLIP_MODEL)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(self.BLIP_MODEL).to(self.device).eval()

        # 3. Translation (EN->ZH)
        self.translation_tokenizer = MarianTokenizer.from_pretrained(self.TRANSLATION_MODEL)
        self.translation_model = MarianMTModel.from_pretrained(self.TRANSLATION_MODEL).to(self.device).eval()

        # 4. PaddleOCR
        use_gpu = (self.device.type == 'cuda')
        self.ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=use_gpu, show_log=False)
```

#### 3.3.2 流水线执行逻辑 (process\_image)

```python
    def process_image(self, image_path: str) -> dict:
        result = {"embedding": None, "caption_en": None, "caption_zh": None, "ocr_text": None, "success": False}

        try:
            image = Image.open(image_path).convert("RGB")

            # 关键：使用 torch.no_grad() 减少内存消耗，提高推理速度
            with torch.no_grad():
                # 1. SigLIP Embedding
                result["embedding"] = self._get_image_embedding(image)

                # 2. BLIP Captioning (EN)
                caption_en = self._get_caption(image)
                result["caption_en"] = caption_en

                # 3. Translation (EN -> ZH)
                if caption_en:
                    result["caption_zh"] = self._translate(caption_en)

            # 4. PaddleOCR
            result["ocr_text"] = self._get_ocr_text(image_path)
            result["success"] = True

        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
        finally:
            # 关键：内存清理 (防止批处理时 GPU OOM)
            self._cleanup_memory()
        
        return result

    def _cleanup_memory(self):
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.empty_cache()
    
    # Helper methods (Implementations for _get_caption, _translate, _get_ocr_text)
    def _get_caption(self, image: Image.Image) -> str:
        inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
        out = self.blip_model.generate(**inputs, max_new_tokens=75)
        return self.blip_processor.decode(out[0], skip_special_tokens=True)

    def _translate(self, text: str) -> str:
        batch = self.translation_tokenizer([text], return_tensors="pt", padding=True).to(self.device)
        generated_ids = self.translation_model.generate(**batch)
        return self.translation_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _get_ocr_text(self, image_path: str) -> str:
        result = self.ocr_engine.ocr(image_path, cls=True)
        if result and result[0]:
            texts = [line[1][0] for line in result[0]]
            return " ".join(texts)
        return ""
```

#### 3.3.3 关键实现细节：SigLIP 编码与归一化

SigLIP 需要用于图像编码（建索引时）和文本编码（搜索时）。两者都**必须**进行 L2 归一化。

```python
    def _normalize(self, embedding):
        # L2 Normalization (关键步骤)
        return torch.nn.functional.normalize(embedding, p=2, dim=1)

    def _get_image_embedding(self, image: Image.Image) -> List[float]:
        # Image Encoding (for indexing)
        inputs = self.siglip_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.siglip_model(**inputs)
        # 使用 pooler_output 获取图像嵌入
        embedding = self._normalize(outputs.pooler_output)
        return embedding.cpu().numpy().flatten().tolist()

    def encode_text_query(self, query: str) -> List[float]:
        # Text Encoding (for search queries)
        with torch.no_grad():
            inputs = self.siglip_processor(text=[query], return_tensors="pt", padding=True, truncation=True).to(self.device)
            # 使用 get_text_features 获取文本嵌入
            embedding = self.siglip_model.get_text_features(**inputs)
            embedding = self._normalize(embedding)
        return embedding.cpu().numpy().flatten().tolist()
```

### 3.4 混合搜索实现 (`app/api/search.py`)

搜索 API 需要执行混合搜索流程。

**流程：**

1.  编码查询文本（获取向量）。
2.  同时执行语义搜索 (ANN) 和关键词搜索 (FTS)。
3.  使用 RRF 融合结果。

#### 3.4.1 搜索执行（SQL/ORM 示例）

```python
# 假设 processor 是 ImageProcessor 实例，db 是数据库会话 (Session)

async def hybrid_search(query: str, limit: int = 50):
    # 1. 编码查询文本
    query_embedding = processor.encode_text_query(query)

    # 2. 执行语义搜索 (ANN) - 使用 SQLAlchemy ORM 方式
    # 使用 pgvector 提供的 cosine_distance 方法进行排序
    semantic_results = db.query(Image).filter(Image.embedding.isnot(None)).order_by(
        Image.embedding.cosine_distance(query_embedding)
    ).limit(limit).all()

    # 3. 执行关键词搜索 (FTS) - 使用 Raw SQL 或 SQLAlchemy Core (text())
    # FTS 功能在 ORM 中支持有限，使用 text() 更直接。
    # 使用 websearch_to_tsquery 更适合处理用户输入的查询。
    fts_sql = text("""
        SELECT id FROM images
        WHERE (to_tsvector('english', coalesce(caption_en, '')) || 
               to_tsvector('simple', coalesce(caption_zh, '') || ' ' || coalesce(ocr_content, ''))) @@ websearch_to_tsquery('simple', :query_text)
        ORDER BY ts_rank(...) DESC -- 可以添加排序逻辑
        LIMIT :limit
    """)
    fts_result_ids = db.execute(fts_sql, {"query_text": query, "limit": limit}).scalars().all()

    # 4. 融合结果 (RRF)
    # 需要将两路结果转换为 RRF 可接受的格式（例如 ID 列表）
    fused_ids = reciprocal_rank_fusion([semantic_results, fts_result_ids])
    
    # 5. (根据 ID 获取完整对象并返回)
    return fused_ids
```

#### 3.4.2 RRF 融合实现

```python
def reciprocal_rank_fusion(search_results_list: list, k=60):
    """
    Applies Reciprocal Rank Fusion (RRF).
    :param search_results_list: A list of lists. Each inner list contains search results sorted by rank.
                                Items can be objects with an 'id' attribute or just IDs.
    :param k: RRF constant (typically 60).
    """
    fused_scores = {}

    for results in search_results_list:
        for rank, item in enumerate(results):
            # 鲁棒性处理：获取 doc_id
            doc_id = item.id if hasattr(item, 'id') else item
            
            # RRF 公式: score = 1 / (k + rank + 1)
            score = 1.0 / (k + rank + 1)
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += score

    # 按融合后的分数排序
    sorted_results_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    
    return sorted_results_ids
```

## IV. 未来演进路线图

PoC1 成功验证后，可以考虑以下增强：

1.  **描述能力增强：** 将 BLIP Base 替换为更强大的 LMM（如本地部署 Qwen-VL 或调用 Gemini/GPT-4o API），以获得更详细的图像描述。
2.  **中文分词优化：** 引入 `pg_jieba` 等 PostgreSQL 中文分词扩展，提升中文关键词搜索的准确率。
