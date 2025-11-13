# 🗺️ AI项目地图 - Vibe Photos开发导航

> 本文档为Coding AI提供项目文档的快速导航地图

## 📚 核心文档体系

### 🎯 开发执行文档
| 文档 | 用途 | 优先级 | 状态 |
|------|------|--------|------|
| [AI_DEVELOPMENT_GUIDE.md](./AI_DEVELOPMENT_GUIDE.md) | 完整开发指南和代码示例 | 🔴 必读 | ✅ 完成 |
| [AI_TASK_TRACKER.md](./AI_TASK_TRACKER.md) | 任务管理和进度跟踪 | 🔴 必读 | ✅ 完成 |
| [AI_CODING_STANDARDS.md](./AI_CODING_STANDARDS.md) | 代码规范和质量标准 | 🔴 必读 | ✅ 完成 |
| [README_FOR_AI.md](./README_FOR_AI.md) | AI快速入门指南 | 🟡 建议 | ✅ 完成 |

### 🏗️ 技术方案文档
| 文档 | 内容 | 使用场景 | 状态 |
|------|------|----------|------|
| [blueprints/AI_BLUEPRINT_GUIDE.md](./blueprints/AI_BLUEPRINT_GUIDE.md) | 架构设计和模块定义 | 实现新模块时参考 | ✅ 新建 |
| [decisions/AI_DECISION_RECORD.md](./decisions/AI_DECISION_RECORD.md) | 技术决策和约束 | 遇到技术选择时查阅 | ✅ 新建 |

## 🚀 开发工作流

### Step 1: 理解项目
```bash
# Read in this order:
1. README_FOR_AI.md          # Quick overview
2. AI_DECISION_RECORD.md     # Technical constraints
3. AI_BLUEPRINT_GUIDE.md     # Architecture design
```

### Step 2: 开始编码
```bash
# Check task and start coding:
1. AI_TASK_TRACKER.md        # Pick a task
2. AI_DEVELOPMENT_GUIDE.md   # Follow examples
3. AI_CODING_STANDARDS.md    # Apply standards
```

### Step 3: 验证质量
```bash
# Verify implementation:
1. Run tests: uv run pytest
2. Check coverage: >80%
3. Update task status in AI_TASK_TRACKER.md
```

## 📋 任务执行优先级

### Phase 1 MVP (Current Focus)
```yaml
priority_order:
  1. Environment Setup:
     - Initialize project structure
     - Configure uv and dependencies
     - Download AI models
     
  2. Core Modules:
     - Image detector (SigLIP + BLIP)
     - Database layer (SQLite)
     - Batch processor
     
  3. API Layer:
     - FastAPI application
     - Core endpoints
     - Error handling
     
  4. Testing:
     - Unit tests
     - Integration tests
     - Performance benchmarks
```

## 🎯 核心技术约束

### Must Follow Rules
```yaml
language:
  code: "English only"           # All source code in English
  docs: "Chinese allowed"         # Documentation can be Chinese
  
technology:
  python: "3.12"                  # Fixed version
  package_manager: "uv"           # No pip/conda/poetry
  
patterns:
  programming: "Functional first" # Avoid unnecessary classes
  errors: "Early return"          # Handle errors early
  async: "Preferred"              # Use async/await for I/O
```

## 📊 文档使用矩阵

| 场景 | 查阅文档 |
|------|----------|
| 开始新任务 | AI_TASK_TRACKER.md |
| 实现新功能 | AI_DEVELOPMENT_GUIDE.md |
| 架构设计 | AI_BLUEPRINT_GUIDE.md |
| 技术选型 | AI_DECISION_RECORD.md |
| 代码规范 | AI_CODING_STANDARDS.md |
| 遇到问题 | AI_DECISION_RECORD.md → Anti-Patterns |
| 性能优化 | AI_BLUEPRINT_GUIDE.md → Performance |
| 测试策略 | AI_CODING_STANDARDS.md → Testing |

## 🔄 项目状态

### Current Phase
```yaml
phase: "Phase 1 MVP"
status: "Ready to implement"
next_milestone: "Core detector module"
```

### Implementation Progress
```yaml
completed:
  - Project documentation ✅
  - Technical decisions ✅
  - Architecture design ✅
  
in_progress:
  - Environment setup 🟡
  
pending:
  - Core modules ⬜
  - API implementation ⬜
  - Testing ⬜
```

## 💡 Quick Commands

### Development Commands
```bash
# Environment setup
uv init
uv add torch transformers fastapi sqlalchemy

# Run development server
uv run uvicorn src.api.main:app --reload

# Run tests
uv run pytest tests/ -v

# Check code quality
uv run ruff check src/
```

### Model Download
```python
# Download required models (run once)
from transformers import AutoModel

AutoModel.from_pretrained("google/siglip-base-patch16-224-i18n")
AutoModel.from_pretrained("Salesforce/blip-image-captioning-base")
```

## 📝 Document Maintenance

### Update Triggers
- Task completion → Update AI_TASK_TRACKER.md
- New technical decision → Update AI_DECISION_RECORD.md
- Architecture change → Update AI_BLUEPRINT_GUIDE.md
- Code pattern discovered → Update AI_CODING_STANDARDS.md

### Version Control
```yaml
commit_format:
  type: ["feat", "fix", "docs", "refactor", "test", "perf"]
  scope: "(module_name)"
  description: "Clear description in English or Chinese"
  
example: "feat(detector): implement SigLIP classification"
```

## ✅ Success Criteria

### Phase 1 Completion
- [ ] All P0 tasks in AI_TASK_TRACKER.md completed
- [ ] Core modules implemented and tested
- [ ] API endpoints functional
- [ ] Test coverage >80%
- [ ] Documentation updated

### Quality Gates
- [ ] No Python code with Chinese comments
- [ ] All functions have type hints
- [ ] All errors properly handled
- [ ] Performance meets requirements
- [ ] Code follows standards

---

**Navigation Guide Version**: 1.0.0
**Project Status**: Ready for Implementation
**Next Action**: Start ENV-001 task from AI_TASK_TRACKER.md
