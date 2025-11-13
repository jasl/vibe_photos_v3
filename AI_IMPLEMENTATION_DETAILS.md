# 📋 AI实施细节补充 - Vibe Photos项目

> 本文档补充AI_AUDIT_REPORT.md中识别的缺失细节，提供具体实施指导

## 🏗️ 项目初始化详细步骤

### ENV-001: 完整目录结构
```bash
#!/bin/bash
# init_project.sh - Complete project initialization script

# Create project structure
mkdir -p vibe_photos_v3
cd vibe_photos_v3

# Source code directories
mkdir -p src/core/{detector,processor,searcher,learner}
mkdir -p src/models/{siglip,blip,ocr,embedder}
mkdir -p src/data/{database,cache,storage}
mkdir -p src/api/{routes,schemas,middleware}
mkdir -p src/utils/{logger,config,metrics}
mkdir -p src/cli

# Test directories
mkdir -p tests/{unit,integration,fixtures/images}
mkdir -p tests/fixtures/{electronic,food,document,landscape,person}

# Configuration and data
mkdir -p config
mkdir -p data
mkdir -p cache/{images/thumbnails,images/processed,detections,ocr,embeddings}
mkdir -p models  # For downloaded AI models
mkdir -p logs
mkdir -p tmp

# Create __init__.py files
find src -type d -exec touch {}/__init__.py \;
find tests -type d -exec touch {}/__init__.py \;

echo "Project structure created successfully!"
```

### ENV-002: Complete pyproject.toml
```toml
[project]
name = "vibe-photos"
version = "1.0.0"
description = "AI-powered photo management system for content creators"
readme = "README.md"
requires-python = ">=3.12"
license = {text = "MIT"}
authors = [
    {name = "Vibe Photos Team", email = "team@vibephotos.ai"}
]

dependencies = [
    # Core dependencies
    "torch==2.9.1",
    "torchvision==0.24.1",
    "transformers==4.57.1",
    "pillow==11.3.0",
    
    # Web framework
    "fastapi==0.121.1",
    "uvicorn[standard]==0.38.0",
    "python-multipart==0.0.6",
    
    # Database
    "sqlalchemy==2.0.44",
    "alembic==1.13.1",
    
    # Data validation
    "pydantic==2.11.10",
    "pydantic-settings==2.2.1",
    
    # OCR
    "paddlepaddle==2.6.0",
    "paddleocr==2.7.3",
    
    # CLI
    "typer==0.20.0",
    "rich==14.2.0",
    
    # Utilities
    "numpy==2.3.4",
    "python-dotenv==1.0.0",
    "aiofiles==23.2.1",
    "httpx==0.25.2",
    "tqdm==4.66.1",
]

[project.optional-dependencies]
dev = [
    "pytest==8.0.0",
    "pytest-asyncio==0.23.2",
    "pytest-cov==4.1.0",
    "ruff==0.6.0",
    "mypy==1.8.0",
    "ipython==8.18.1",
    "notebook==7.0.6",
]

phase2 = [
    "sentence-transformers==2.5.1",
    "faiss-cpu==1.7.4",
]

phase3 = [
    "psycopg2-binary==2.9.9",
    "redis==5.0.1",
    "celery==5.3.4",
    "prometheus-client==0.19.0",
]

[project.scripts]
vibe = "src.cli:app"
vibe-server = "src.api.main:run"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = []

[tool.uv.sources]
torch = [
    {index = "pytorch-cuda", marker = "sys_platform == 'linux' or sys_platform == 'win32'"},
]
torchvision = [
    {index = "pytorch-cuda", marker = "sys_platform == 'linux' or sys_platform == 'win32'"},
]

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "ARG", "ERA"]
ignore = ["E501"]

[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-ra -q --strict-markers --cov=src --cov-report=term-missing"

[tool.mypy]
python_version = "3.12"
warn_return_any = true
disallow_untyped_defs = true
```

## 💾 数据库完整Schema

### DB-001: Complete SQLite Schema
```sql
-- File: config/schema.sql

-- Main photos table
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- File information
    path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    hash VARCHAR(64) UNIQUE NOT NULL,  -- MD5 or SHA256
    
    -- Image metadata
    width INTEGER,
    height INTEGER,
    size_bytes BIGINT,
    format VARCHAR(10),  -- jpg, png, etc.
    
    -- Timestamps
    taken_at TIMESTAMP,  -- EXIF date if available
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- AI detection results
    category VARCHAR(100),  -- Primary category
    confidence REAL CHECK(confidence >= 0 AND confidence <= 1),
    classifications JSON,  -- All classification results
    caption TEXT,  -- BLIP-generated caption
    ocr_text TEXT,  -- Extracted text
    detections JSON,  -- Object detection results [{class, bbox, score}]
    
    -- Embeddings (Phase 2)
    embedding_json TEXT,  -- JSON-serialized vector
    
    -- User data
    user_label VARCHAR(200),
    user_tags TEXT,  -- Comma-separated tags
    user_notes TEXT,
    is_favorite BOOLEAN DEFAULT 0,
    is_hidden BOOLEAN DEFAULT 0,
    
    -- Processing status
    status VARCHAR(20) DEFAULT 'pending',  -- pending, processing, completed, error
    error_message TEXT
);

-- Indexes for performance
CREATE INDEX idx_photos_category ON photos(category);
CREATE INDEX idx_photos_confidence ON photos(confidence);
CREATE INDEX idx_photos_status ON photos(status);
CREATE INDEX idx_photos_imported_at ON photos(imported_at);
CREATE INDEX idx_photos_user_label ON photos(user_label);
CREATE INDEX idx_photos_hash ON photos(hash);

-- Full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS photos_fts USING fts5(
    photo_id,
    caption,
    ocr_text,
    user_label,
    user_tags,
    content=photos,
    content_rowid=id
);

-- Triggers to keep FTS in sync
CREATE TRIGGER photos_ai AFTER INSERT ON photos BEGIN
    INSERT INTO photos_fts(photo_id, caption, ocr_text, user_label, user_tags)
    VALUES (new.id, new.caption, new.ocr_text, new.user_label, new.user_tags);
END;

CREATE TRIGGER photos_ad AFTER DELETE ON photos BEGIN
    DELETE FROM photos_fts WHERE photo_id = old.id;
END;

CREATE TRIGGER photos_au AFTER UPDATE ON photos BEGIN
    UPDATE photos_fts 
    SET caption = new.caption,
        ocr_text = new.ocr_text,
        user_label = new.user_label,
        user_tags = new.user_tags
    WHERE photo_id = new.id;
END;

-- Annotation history table
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    
    ai_prediction VARCHAR(100),
    user_correction VARCHAR(100),
    confidence REAL,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(50) DEFAULT 'system'
);

-- Processing queue table
CREATE TABLE IF NOT EXISTS processing_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER REFERENCES photos(id) ON DELETE CASCADE,
    
    task_type VARCHAR(50) NOT NULL,  -- detect, ocr, embed, etc.
    priority INTEGER DEFAULT 5,
    status VARCHAR(20) DEFAULT 'pending',
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    result JSON,
    error_message TEXT
);

-- Search history (for analytics)
CREATE TABLE IF NOT EXISTS search_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    result_count INTEGER,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🔧 模型管理策略

### ENV-004: Model Download Script
```python
#!/usr/bin/env python
# File: scripts/download_models.py

import os
from pathlib import Path
from transformers import AutoModel, AutoProcessor, BlipProcessor, BlipForConditionalGeneration
import torch

def setup_model_cache():
    """Set up model cache directory"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Set environment variable
    os.environ["TRANSFORMERS_CACHE"] = str(models_dir.absolute())
    os.environ["HF_HOME"] = str(models_dir.absolute())
    
    return models_dir

def download_siglip():
    """Download SigLIP model"""
    print("Downloading SigLIP model...")
    model_name = "google/siglip-base-patch16-224-i18n"
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    print(f"✅ SigLIP downloaded: {model_name}")
    return model, processor

def download_blip():
    """Download BLIP model"""
    print("Downloading BLIP model...")
    model_name = "Salesforce/blip-image-captioning-base"
    
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    
    print(f"✅ BLIP downloaded: {model_name}")
    return model, processor

def verify_models():
    """Verify models are working"""
    print("\nVerifying models...")
    
    # Test image (create a dummy one)
    from PIL import Image
    import numpy as np
    
    dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    
    # Test SigLIP
    siglip_model, siglip_processor = download_siglip()
    inputs = siglip_processor(
        text=["test"],
        images=dummy_image,
        return_tensors="pt",
        padding=True
    )
    print("✅ SigLIP model verified")
    
    # Test BLIP
    blip_model, blip_processor = download_blip()
    inputs = blip_processor(dummy_image, return_tensors="pt")
    print("✅ BLIP model verified")
    
    print("\n✨ All models downloaded and verified successfully!")

if __name__ == "__main__":
    models_dir = setup_model_cache()
    print(f"Models will be saved to: {models_dir.absolute()}")
    
    download_siglip()
    download_blip()
    verify_models()
```

## 📡 API端点详细规格

### API-002: /import/batch Endpoint
```python
# File: src/api/schemas/import_schemas.py

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from pathlib import Path

class BatchImportRequest(BaseModel):
    """Batch import request schema"""
    directory: str = Field(..., description="Directory path containing images")
    recursive: bool = Field(True, description="Scan subdirectories")
    extensions: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".webp"],
        description="Image file extensions to process"
    )
    skip_processed: bool = Field(True, description="Skip already processed images")
    max_workers: int = Field(4, ge=1, le=16, description="Parallel processing threads")
    
    @validator('directory')
    def validate_directory(cls, v):
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Directory does not exist: {v}")
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {v}")
        return str(path.absolute())

class BatchImportResponse(BaseModel):
    """Batch import response schema"""
    task_id: str = Field(..., description="Async task ID")
    total_files: int = Field(..., description="Total files found")
    queued: int = Field(..., description="Files queued for processing")
    skipped: int = Field(..., description="Files skipped (already processed)")
    status: str = Field(..., description="Task status")
    
class ImportProgress(BaseModel):
    """Import progress schema"""
    task_id: str
    total: int
    processed: int
    failed: int
    current_file: Optional[str]
    percentage: float = Field(..., ge=0, le=100)
    estimated_time_remaining: Optional[int]  # seconds
```

### API-003: /search Endpoint
```python
# File: src/api/schemas/search_schemas.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class SearchRequest(BaseModel):
    """Search request schema"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    
    # Filters
    categories: Optional[List[str]] = Field(None, description="Filter by categories")
    min_confidence: float = Field(0.5, ge=0, le=1, description="Minimum confidence")
    date_from: Optional[datetime] = Field(None, description="Start date filter")
    date_to: Optional[datetime] = Field(None, description="End date filter")
    
    # Search options
    search_mode: str = Field("text", pattern="^(text|vector|hybrid)$")
    include_ocr: bool = Field(True, description="Search in OCR text")
    include_captions: bool = Field(True, description="Search in captions")
    
    # Pagination
    offset: int = Field(0, ge=0, description="Results offset")
    limit: int = Field(20, ge=1, le=100, description="Results limit")
    
class PhotoResult(BaseModel):
    """Single photo search result"""
    id: int
    path: str
    thumbnail_url: Optional[str]
    
    # Metadata
    category: str
    confidence: float
    caption: Optional[str]
    ocr_text: Optional[str]
    
    # User data
    user_label: Optional[str]
    user_tags: List[str] = []
    
    # Relevance
    score: float = Field(..., description="Search relevance score")
    match_reason: str = Field(..., description="Why this result matched")

class SearchResponse(BaseModel):
    """Search response schema"""
    query: str
    total_results: int
    returned_results: int
    search_time_ms: int
    results: List[PhotoResult]
    
    # Facets for filtering
    facets: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Available filters with counts"
    )
```

## 📁 测试数据准备

### Test Fixtures Setup
```python
#!/usr/bin/env python
# File: scripts/prepare_test_data.py

import os
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_test_images():
    """Create synthetic test images for each category"""
    fixtures_dir = Path("tests/fixtures/images")
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    
    categories = {
        "electronic": ["iPhone", "MacBook", "Camera"],
        "food": ["Pizza", "Coffee", "Cake"],
        "document": ["Invoice", "Receipt", "Report"],
        "landscape": ["Mountain", "Beach", "Forest"],
        "person": ["Portrait", "Group", "Selfie"]
    }
    
    for category, items in categories.items():
        category_dir = fixtures_dir / category
        category_dir.mkdir(exist_ok=True)
        
        for item in items:
            # Create a simple test image with text
            img = Image.new('RGB', (224, 224), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # Add category and item text
            draw.text((10, 10), f"{category.upper()}", fill=(0, 0, 0))
            draw.text((10, 100), item, fill=(0, 0, 0))
            
            # Add some random pixels for variation
            pixels = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            patch = Image.fromarray(pixels)
            img.paste(patch, (87, 87))
            
            # Save image
            filename = category_dir / f"{item.lower().replace(' ', '_')}.jpg"
            img.save(filename, quality=95)
            print(f"Created: {filename}")
    
    print(f"\n✅ Test images created in {fixtures_dir}")

def create_mock_data():
    """Create mock data for testing"""
    mock_file = Path("tests/fixtures/mock_data.json")
    
    mock_data = {
        "sample_detections": [
            {
                "category": "electronic",
                "confidence": 0.95,
                "caption": "A smartphone on a wooden desk"
            },
            {
                "category": "food",
                "confidence": 0.88,
                "caption": "A pizza with various toppings"
            }
        ],
        "sample_ocr": [
            {
                "text": "Invoice #12345",
                "confidence": 0.92
            }
        ]
    }
    
    import json
    with open(mock_file, 'w') as f:
        json.dump(mock_data, f, indent=2)
    
    print(f"✅ Mock data created: {mock_file}")

if __name__ == "__main__":
    create_test_images()
    create_mock_data()
```

## 🔧 配置文件模板

### Complete Configuration Template
```yaml
# File: config/settings.yaml

# Application settings
app:
  name: "Vibe Photos"
  version: "1.0.0"
  debug: false
  log_level: "INFO"

# API settings
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  cors_enabled: true
  cors_origins: ["*"]
  max_upload_size: 104857600  # 100MB
  rate_limit: "100/minute"

# Database settings
database:
  url: "sqlite:///data/vibe_photos.db"
  echo: false
  pool_size: 5
  max_overflow: 10

# Model settings
models:
  device: "cuda:0"  # cuda:0, cpu, mps
  cache_dir: "./models"
  
  siglip:
    name: "google/siglip-base-patch16-224-i18n"
    batch_size: 16
    
  blip:
    name: "Salesforce/blip-image-captioning-base"
    max_caption_length: 50
    
  ocr:
    lang: "ch"  # ch, en
    use_angle_cls: true
    use_gpu: false

# Processing settings
processing:
  batch_size: 32
  max_workers: 4
  thumbnail_size: [224, 224]
  supported_formats: [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
  
  # Quality thresholds
  confidence_threshold: 0.5
  ocr_confidence_threshold: 0.7

# Cache settings
cache:
  enabled: true
  ttl: 3600  # 1 hour
  max_size: 1073741824  # 1GB
  directory: "./cache"

# Logging settings
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./logs/vibe_photos.log"
  max_bytes: 10485760  # 10MB
  backup_count: 5

# Feature flags
features:
  ocr_enabled: true
  embedding_enabled: false  # Phase 2
  few_shot_learning: false  # Phase 3
  auto_tagging: true
  duplicate_detection: true
```

## ✅ 启动检查清单

### Pre-flight Checklist
```python
#!/usr/bin/env python
# File: scripts/preflight_check.py

import sys
from pathlib import Path
import subprocess

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major != 3 or version.minor < 12:
        print(f"❌ Python 3.12+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} detected")
    return True

def check_uv_installed():
    """Check if uv is installed"""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True)
        print(f"✅ uv is installed")
        return True
    except FileNotFoundError:
        print("❌ uv is not installed")
        print("   Install with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False

def check_directories():
    """Check required directories exist"""
    required_dirs = [
        "src", "tests", "config", "data", "cache", "models", "logs"
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ Directory exists: {dir_name}")
        else:
            print(f"❌ Missing directory: {dir_name}")
            all_exist = False
    
    return all_exist

def check_config_files():
    """Check configuration files"""
    required_files = [
        "pyproject.toml",
        "config/settings.yaml",
        "config/schema.sql"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ Config file exists: {file_path}")
        else:
            print(f"❌ Missing config file: {file_path}")
            all_exist = False
    
    return all_exist

def main():
    """Run all checks"""
    print("🚀 Running preflight checks...\n")
    
    checks = [
        check_python_version(),
        check_uv_installed(),
        check_directories(),
        check_config_files()
    ]
    
    if all(checks):
        print("\n✨ All checks passed! Ready to start development.")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

---

**文档版本**: 1.0.0
**创建日期**: 2024-11-12
**用途**: 补充实施细节，解决AI_AUDIT_REPORT中的问题
