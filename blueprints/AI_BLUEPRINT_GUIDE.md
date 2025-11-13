# 🏗️ AI蓝图指南 - Vibe Photos技术实施方案

> 本文档为Coding AI提供结构化的技术实施方案，定义了系统的架构、模块划分和实现细节

## 📋 系统架构定义

### 核心架构模式
```yaml
architecture_pattern: "Progressive Monolith"  # Start simple, evolve as needed
deployment_model: "Local First"              # Local deployment, cloud ready
data_flow: "Pipeline Pattern"                # Sequential processing pipeline
api_style: "RESTful + Async"                 # FastAPI async endpoints
```

### 系统模块划分

```
vibe_photos/
├── core/               # Core business logic
│   ├── detector/      # Image detection module
│   ├── processor/     # Batch processing module
│   ├── searcher/      # Search engine module
│   └── learner/       # Learning module (Phase 2+)
│
├── models/            # AI model wrappers
│   ├── siglip/       # SigLIP model integration
│   ├── blip/         # BLIP model integration
│   ├── ocr/          # OCR model integration
│   └── embedder/     # Vector embedding (Phase 2+)
│
├── data/             # Data layer
│   ├── database/     # Database operations
│   ├── cache/        # Cache management
│   └── storage/      # File storage
│
├── api/              # API layer
│   ├── routes/       # API routes
│   ├── schemas/      # Pydantic models
│   └── middleware/   # Custom middleware
│
└── utils/            # Utilities
    ├── logger/       # Logging configuration
    ├── config/       # Configuration management
    └── metrics/      # Performance metrics
```

## 🎯 Phase 1: MVP Implementation

### 必须实现的模块

#### 1. Image Detector Module
```python
# Path: src/core/detector/image_detector.py
class ImageDetector:
    """
    Core image detection using SigLIP + BLIP
    
    Requirements:
    - Multi-language classification (Chinese support)
    - Zero-shot learning capability
    - Caption generation
    - Confidence scoring
    """
    
    def __init__(self):
        # Load SigLIP for classification
        # Load BLIP for captioning
        pass
    
    def detect(self, image_path: Path) -> DetectionResult:
        # 1. Validate image
        # 2. Run SigLIP classification
        # 3. Generate BLIP caption
        # 4. Merge results
        # 5. Return structured result
        pass
```

#### 2. Batch Processor Module
```python
# Path: src/core/processor/batch_processor.py
class BatchProcessor:
    """
    High-performance batch processing
    
    Requirements:
    - Process >10 images/second
    - Incremental processing (skip processed)
    - Parallel execution
    - Progress tracking
    - Error recovery
    """
    
    def process_directory(self, path: Path) -> ProcessingResult:
        # 1. Scan for images
        # 2. Filter already processed
        # 3. Create processing pool
        # 4. Execute in parallel
        # 5. Handle errors gracefully
        # 6. Return statistics
        pass
```

#### 3. Database Layer
```python
# Path: src/data/database/manager.py
class DatabaseManager:
    """
    SQLite database operations
    
    Schema:
    - photos table (main data)
    - annotations table (user labels)
    - search_logs table (analytics)
    
    Requirements:
    - CRUD operations
    - Full-text search
    - Batch insert optimization
    - Connection pooling
    """
    
    def initialize(self):
        # Create tables if not exist
        pass
    
    def insert_photo(self, photo_data: PhotoData) -> int:
        # Insert with deduplication check
        pass
    
    def search(self, query: str, filters: SearchFilters) -> List[Photo]:
        # Execute optimized search
        pass
```

#### 4. FastAPI Application
```python
# Path: src/api/main.py
app = FastAPI(
    title="Vibe Photos API",
    version="1.0.0"
)

# Required endpoints:
# POST /import/batch    - Batch import photos
# POST /import/single   - Single photo upload
# GET  /search         - Search photos
# POST /annotate      - Add user annotations
# GET  /stats         - System statistics
# GET  /health        - Health check
```

### 实现优先级

| Priority | Module | Dependencies | Output |
|----------|--------|--------------|--------|
| P0 | Image Detector | - | Detection capability |
| P0 | Database Layer | - | Data persistence |
| P1 | Batch Processor | Image Detector, Database | Bulk processing |
| P1 | FastAPI Routes | All above | API access |
| P2 | CLI Interface | Core modules | Command line tool |
| P2 | Caching Layer | Database | Performance optimization |

### 质量标准

```yaml
performance:
  processing_speed: ">10 images/second"
  single_image_time: "<2 seconds"
  search_response: "<500ms"
  memory_usage: "<2GB"

accuracy:
  classification: ">85%"
  ocr_chinese: ">90%"
  duplicate_detection: "100%"

code_quality:
  test_coverage: ">80%"
  type_hints: "100%"
  documentation: "All public methods"
  error_handling: "All I/O operations"
```

## 🚀 Phase 2: Enhanced Features

### 新增模块

#### 1. Vector Embedder
```python
# Path: src/models/embedder/vector_embedder.py
class VectorEmbedder:
    """
    Generate vector embeddings for semantic search
    
    Requirements:
    - Image to vector encoding
    - Text to vector encoding
    - Similarity computation
    - Batch encoding support
    """
```

#### 2. Hybrid Searcher
```python
# Path: src/core/searcher/hybrid_searcher.py
class HybridSearcher:
    """
    Combine text and vector search
    
    Requirements:
    - Text search (SQLite FTS)
    - Vector search (cosine similarity)
    - Result fusion (RRF algorithm)
    - Ranking optimization
    """
```

### 架构升级

```yaml
changes:
  - Add vector storage to database (JSON field)
  - Implement embedding pipeline
  - Add similarity search endpoint
  - Cache embeddings for performance
```

## 💎 Phase 3: Production System

### 架构迁移

#### PostgreSQL + pgvector Setup
```sql
-- Required schema changes
CREATE EXTENSION vector;

CREATE TABLE photos_v2 (
    id SERIAL PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    embedding vector(768),  -- SigLIP embedding dimension
    -- ... other fields
);

CREATE INDEX ON photos_v2 USING hnsw (embedding vector_cosine_ops);
```

#### Celery Task Queue
```python
# Path: src/tasks/processing_tasks.py
from celery import Celery

app = Celery('vibe_photos')

@app.task
def process_image_async(image_path: str):
    """Async image processing task"""
    pass

@app.task
def generate_embedding_async(photo_id: int):
    """Async embedding generation"""
    pass
```

### 生产部署架构

```yaml
components:
  api_server:
    framework: "FastAPI"
    workers: 4
    deployment: "Docker"
  
  database:
    primary: "PostgreSQL 15"
    extensions: ["pgvector"]
    connection_pool: 20
  
  cache:
    service: "Redis"
    ttl: 3600
    
  queue:
    broker: "Redis"
    worker: "Celery"
    concurrency: 4
    
  storage:
    images: "Local FS / S3"
    models: "Local cache"
```

## 📦 模型配置

### Required Models

```yaml
phase1:
  siglip:
    model: "google/siglip-base-patch16-224-i18n"
    size: "~400MB"
    purpose: "Multi-language classification"
  
  blip:
    model: "Salesforce/blip-image-captioning-base"
    size: "~990MB"
    purpose: "Image captioning"

phase2:
  # Add to phase1 models
  embedder:
    model: "google/siglip-base-patch16-224"
    size: "~400MB"
    purpose: "Vector embeddings"

phase3:
  # Upgrade models
  siglip:
    model: "google/siglip-large-patch16-384-i18n"
    size: "~1.5GB"
    purpose: "Higher accuracy"
  
  ocr:
    model: "PaddleOCR PP-OCRv4"
    size: "~200MB"
    purpose: "Text extraction"
```

## 🔧 Implementation Patterns

### Error Handling Pattern
```python
from typing import Optional, Union
from dataclasses import dataclass

@dataclass
class Result[T]:
    """Result wrapper for error handling"""
    value: Optional[T] = None
    error: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        return self.error is None
    
def safe_operation() -> Result[str]:
    try:
        # Operation logic
        return Result(value="success")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return Result(error=str(e))
```

### Async Processing Pattern
```python
import asyncio
from typing import List

async def process_batch_async(items: List[Path]) -> List[Result]:
    """Process items concurrently"""
    tasks = [process_item_async(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter successful results
    return [r for r in results if not isinstance(r, Exception)]
```

### Caching Pattern
```python
from functools import lru_cache
import hashlib

class CacheManager:
    def __init__(self):
        self._cache = {}
    
    def get_or_compute(self, key: str, compute_fn):
        """Get from cache or compute"""
        if key not in self._cache:
            self._cache[key] = compute_fn()
        return self._cache[key]
    
    @staticmethod
    def make_key(image_path: Path) -> str:
        """Generate cache key from file content"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
```

## 📊 数据流设计

### Import Pipeline
```
Input Image → Validation → Hash Computation → Duplicate Check
    ↓
Detection (SigLIP + BLIP) → OCR (if needed) → Database Insert
    ↓
Embedding Generation (Phase 2+) → Index Update → Cache Update
```

### Search Pipeline
```
Search Query → Query Parser → Text Search ↘
                                            → Result Fusion → Ranking → Response
Vector Encoding → Vector Search ↗
```

## ✅ Implementation Checklist

### Phase 1 Deliverables
- [ ] Core detection module with SigLIP + BLIP
- [ ] SQLite database with schema
- [ ] Batch processing with parallelization
- [ ] FastAPI with 6 core endpoints
- [ ] CLI tool for import/search
- [ ] Unit tests with >80% coverage
- [ ] API documentation

### Phase 2 Deliverables
- [ ] Vector embedding module
- [ ] Hybrid search implementation
- [ ] Embedding storage in database
- [ ] Search fusion algorithm
- [ ] Performance optimizations
- [ ] Enhanced API endpoints

### Phase 3 Deliverables
- [ ] PostgreSQL migration
- [ ] pgvector integration
- [ ] Celery task queue
- [ ] Redis caching
- [ ] Docker deployment
- [ ] Production monitoring
- [ ] Scalability testing

## 🎯 Success Metrics

```yaml
functional:
  image_types_supported: ["electronic", "food", "document", "person", "landscape"]
  languages_supported: ["en", "zh", "ja", "ko"]
  search_modes: ["text", "semantic", "hybrid"]
  
performance:
  import_throughput: ">100 images/minute"
  search_latency_p95: "<500ms"
  concurrent_users: ">100"
  
reliability:
  uptime: ">99.5%"
  data_integrity: "100%"
  error_recovery: "Automatic"
```

---

**Document Type**: Technical Blueprint
**Target Audience**: Coding AI
**Language Rule**: All code and comments in English
**Version**: 1.0.0
