# 🎯 AI决策记录 - Vibe Photos技术决策指南

> 本文档为Coding AI提供所有技术决策的结构化记录，每个决策都包含明确的执行指令

## 📋 决策执行规则

### 决策优先级
- 🔴 **MUST** - 必须执行，不可更改
- 🟡 **SHOULD** - 应该执行，特殊情况可调整
- 🟢 **MAY** - 可选执行，根据情况决定

### 决策状态
- ✅ **ACTIVE** - 当前有效，必须遵循
- ⚠️ **DEPRECATED** - 已废弃，使用替代方案
- 🔄 **PENDING** - 待确认，暂不执行

## 🏗️ 架构决策

### ARC-001: Progressive Monolith Architecture
**Priority**: 🔴 MUST  
**Status**: ✅ ACTIVE  
**Decision**: Start with monolithic architecture, evolve to microservices only when needed

```yaml
implementation:
  phase1:
    pattern: "Single application"
    deployment: "Single process"
    database: "Single SQLite file"
    
  phase2:
    pattern: "Modular monolith"
    deployment: "Single process with workers"
    database: "SQLite with JSON fields"
    
  phase3:
    pattern: "Service-oriented"
    deployment: "Multiple processes"
    database: "PostgreSQL + pgvector"
    
constraints:
  - No premature optimization
  - No unnecessary abstractions
  - Keep it simple until proven otherwise
```

### ARC-002: Offline-First Processing
**Priority**: 🔴 MUST  
**Status**: ✅ ACTIVE  
**Decision**: Process images in batch offline mode, not real-time

```yaml
rationale:
  - Reduces system complexity
  - Better error recovery
  - Predictable performance
  
implementation:
  - Use batch processing for import
  - Queue processing for large datasets
  - Async task execution for heavy operations
  
exceptions:
  - Search must be real-time (<500ms)
  - UI preview generation can be real-time
```

## 💻 Technology Stack Decisions

### TECH-001: Python 3.12 + uv
**Priority**: 🔴 MUST  
**Status**: ✅ ACTIVE  
**Decision**: Use Python 3.12 with uv package manager exclusively

```yaml
requirements:
  python_version: "3.12"
  package_manager: "uv"
  
forbidden:
  - pip
  - poetry
  - conda
  - pipenv
  
commands:
  install: "uv add <package>"
  sync: "uv sync"
  run: "uv run <command>"
```

### TECH-002: FastAPI for API Layer
**Priority**: 🔴 MUST  
**Status**: ✅ ACTIVE  
**Decision**: Use FastAPI for all API endpoints

```yaml
implementation:
  - Async endpoints by default
  - Pydantic for validation
  - Automatic OpenAPI documentation
  - Type hints for all parameters
  
patterns:
  - Dependency injection for database
  - Background tasks for heavy operations
  - Middleware for logging and auth
```

### TECH-003: SQLite → PostgreSQL Migration Path
**Priority**: 🟡 SHOULD  
**Status**: ✅ ACTIVE  
**Decision**: Start with SQLite, migrate to PostgreSQL when scaling

```yaml
trigger_conditions:
  - Data size > 10GB
  - Concurrent users > 50
  - Need vector search capability
  
migration_strategy:
  1. Keep SQLite for Phase 1 MVP
  2. Add migration scripts early
  3. Test PostgreSQL in Phase 2
  4. Deploy PostgreSQL in Phase 3
```

## 🤖 AI Model Decisions

### MODEL-001: SigLIP + BLIP Combination
**Priority**: 🔴 MUST  
**Status**: ✅ ACTIVE  
**Decision**: Use SigLIP for classification and BLIP for captioning

```yaml
models:
  classification:
    name: "google/siglip-base-patch16-224-i18n"
    purpose: "Multi-language zero-shot classification"
    size: "~400MB"
    
  captioning:
    name: "Salesforce/blip-image-captioning-base"
    purpose: "Generate natural language descriptions"
    size: "~990MB"
    
implementation:
  - Load models on startup
  - Cache in memory
  - Use batch processing when possible
  
performance:
  - Expected accuracy: >85%
  - Processing time: <2s per image
```

### MODEL-002: Abandon RTMDet
**Priority**: 🔴 MUST  
**Status**: ⚠️ DEPRECATED  
**Decision**: Do NOT use RTMDet due to dependency issues

```yaml
problems:
  - mmcv incompatible with Python 3.11+
  - Complex dependency chain
  - Poor documentation
  
replacement:
  use: "SigLIP + BLIP"
  reason: "Better compatibility and performance"
```

### MODEL-003: PaddleOCR for Text Extraction
**Priority**: 🟡 SHOULD  
**Status**: ✅ ACTIVE  
**Decision**: Use PaddleOCR for Chinese text extraction

```yaml
model:
  name: "PaddleOCR PP-OCRv4"
  languages: ["ch", "en"]
  size: "~200MB"
  
when_to_use:
  - Document images
  - Screenshots
  - Images with visible text
  
configuration:
  use_angle_cls: true
  use_gpu: false  # CPU by default
  lang: "ch"      # Chinese + English
```

## 📊 Data Layer Decisions

### DATA-001: Embedding Storage Strategy
**Priority**: 🟡 SHOULD  
**Status**: ✅ ACTIVE  
**Decision**: Store embeddings as JSON in Phase 1-2, migrate to pgvector in Phase 3

```yaml
phase1:
  storage: "JSON string in TEXT column"
  format: "Base64 encoded numpy array"
  
phase2:
  storage: "JSON field in SQLite"
  indexing: "In-memory numpy arrays"
  
phase3:
  storage: "pgvector native type"
  indexing: "HNSW index"
  dimension: 768  # SigLIP embedding size
```

### DATA-002: Caching Strategy
**Priority**: 🟡 SHOULD  
**Status**: ✅ ACTIVE  
**Decision**: Implement multi-level caching

```yaml
levels:
  L1_memory:
    tool: "Python dict / lru_cache"
    ttl: "5 minutes"
    size: "100MB"
    
  L2_disk:
    tool: "File system"
    ttl: "1 hour"
    size: "1GB"
    
  L3_distributed:  # Phase 3 only
    tool: "Redis"
    ttl: "24 hours"
    size: "Unlimited"
    
what_to_cache:
  - Model predictions
  - Thumbnails
  - Search results
  - Embeddings
```

## 🔐 Development Practice Decisions

### DEV-001: Code Language Policy
**Priority**: 🔴 MUST  
**Status**: ✅ ACTIVE  
**Decision**: All source code in English, documentation in Chinese

```yaml
english_required:
  - Source code files (.py, .js, .yaml)
  - Code comments
  - Docstrings
  - Log messages
  - Variable/function names
  - Git commit types (feat:, fix:, etc.)
  
chinese_allowed:
  - Documentation files (.md)
  - User-facing messages (localized)
  - Git commit descriptions
```

### DEV-002: Testing Strategy
**Priority**: 🔴 MUST  
**Status**: ✅ ACTIVE  
**Decision**: Test-Driven Development with >80% coverage

```yaml
requirements:
  - Write tests before implementation
  - Unit tests for all public methods
  - Integration tests for API endpoints
  - Coverage target: >80%
  
tools:
  framework: "pytest"
  coverage: "pytest-cov"
  mocking: "unittest.mock"
  async: "pytest-asyncio"
```

### DEV-003: Error Handling Pattern
**Priority**: 🔴 MUST  
**Status**: ✅ ACTIVE  
**Decision**: Use early return pattern with Result types

```python
# Required pattern:
def operation() -> Result[T]:
    # Validate inputs first
    if not valid:
        return Result(error="Validation failed")
    
    # Try operation
    try:
        value = perform_operation()
        return Result(value=value)
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return Result(error=str(e))
```

## 🚀 Deployment Decisions

### DEPLOY-001: Container Strategy
**Priority**: 🟢 MAY  
**Status**: 🔄 PENDING  
**Decision**: Containerize only for production deployment

```yaml
phase1:
  deployment: "Direct Python execution"
  reason: "Faster development iteration"
  
phase3:
  deployment: "Docker containers"
  orchestration: "docker-compose or K8s"
  registry: "Local or DockerHub"
```

### DEPLOY-002: Monitoring Strategy
**Priority**: 🟢 MAY  
**Status**: 🔄 PENDING  
**Decision**: Add monitoring only in production

```yaml
metrics:
  - Response time
  - Error rate
  - Processing throughput
  - Memory usage
  
tools:
  phase1: "Python logging only"
  phase3:
    metrics: "Prometheus"
    visualization: "Grafana"
    tracing: "OpenTelemetry"
```

## ⚠️ Anti-Patterns to Avoid

### What NOT to Do

```yaml
avoid:
  - Premature optimization
  - Over-engineering
  - Deep inheritance hierarchies
  - Global state
  - Synchronous blocking operations
  - Hardcoded configurations
  - Mixed language in source code
  - Untested code
  - Poor error messages
  - Memory leaks
  
specifically_forbidden:
  - Using pip instead of uv
  - Python < 3.12
  - Class-based views in FastAPI
  - Storing large blobs in database
  - Real-time processing requirements
  - Distributed transactions
```

## 📈 Decision Evaluation Criteria

### When to Override a Decision

```yaml
override_conditions:
  - Performance degradation >50%
  - Security vulnerability discovered
  - Dependency becomes unmaintained
  - Better alternative with 10x improvement
  
override_process:
  1. Document the problem
  2. Propose alternative
  3. Test alternative
  4. Update this document
  5. Migrate gradually
```

## 🔄 Decision Review Triggers

```yaml
review_triggers:
  user_growth:
    threshold: ">1000 active users"
    action: "Review architecture decisions"
    
  data_size:
    threshold: ">1TB photos"
    action: "Review storage decisions"
    
  performance:
    threshold: "Response time >1s"
    action: "Review optimization decisions"
    
  error_rate:
    threshold: ">1% requests failing"
    action: "Review reliability decisions"
```

## ✅ Implementation Priority

### Decision Execution Order

1. **Foundation** (Must complete first)
   - TECH-001: Python 3.12 + uv
   - DEV-001: Code language policy
   - MODEL-001: SigLIP + BLIP

2. **Core** (Required for MVP)
   - ARC-001: Progressive monolith
   - TECH-002: FastAPI
   - DATA-001: Embedding storage
   - DEV-002: Testing strategy

3. **Enhancement** (Can defer to Phase 2)
   - MODEL-003: PaddleOCR
   - DATA-002: Caching strategy
   - TECH-003: PostgreSQL migration

4. **Production** (Defer to Phase 3)
   - DEPLOY-001: Containerization
   - DEPLOY-002: Monitoring

---

**Document Type**: Technical Decision Record
**Target Audience**: Coding AI
**Update Frequency**: On significant changes only
**Version**: 1.0.0
