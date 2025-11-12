# Vibe Photos - 项目根目录

## 📁 项目结构

```
vibe_photos_v3/
├── blueprints/           # 📁 所有设计文档（统一管理）
│   ├── phase1/          # Phase 1：基础验证
│   ├── phase2/          # Phase 2：功能增强
│   └── phase_final/     # Phase Final：完整系统
├── decisions/            # 🎯 技术决策中心（独立管理）
│   ├── TECHNICAL_DECISIONS.md  # 综合技术决策
│   └── archives/        # 历史决策归档
├── pyproject.toml        # Python 3.12 项目配置
├── .python-version       # Python 版本固定
├── POC_PHASE_NOTICE.md   # ⚠️ POC阶段重要说明
├── ROADMAP.md            # 🚀 完整产品路线图
├── FINAL_CHECKLIST.md    # ✅ 最终文档检查清单
├── UV_USAGE.md           # 📦 Python环境管理规范（必读）
├── DIRECTORY_STRUCTURE.md # 📂 目录使用说明
├── samples/              # 原始测试数据集（只读）
├── data/                 # 处理结果存储（读写）
├── cache/                # 可复用缓存（跨版本共享）
├── models/               # 预训练模型（首次下载后复用）
├── log/                  # 运行日志（自动轮转）
├── tmp/                  # 临时文件（运行期间）
├── DEPENDENCIES.md       # 所有依赖版本清单
└── LICENSE              # 项目许可证
```

详细目录说明请查看 [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md)

## 🎯 快速导航

### ⚠️ Python环境管理（必读）
- **Python版本**：3.12（Ubuntu 24.04 LTS 默认）
- **强制要求**：统一使用 `uv` 管理Python环境
- **禁止使用**：pip/poetry/conda等其他工具
- **使用指南**：[UV_USAGE.md](UV_USAGE.md)

### 🎯 技术决策中心
- **综合决策文档**：[decisions/TECHNICAL_DECISIONS.md](decisions/TECHNICAL_DECISIONS.md)
- **决策索引**：[decisions/README.md](decisions/README.md)
- **历史归档**：[decisions/archives/](decisions/archives/)

### 📄 产品路线图
- **完整规划**：Phase 1 → Phase 2 → Phase Final
- **时间线**：3-6个月渐进式升级
- **文档**：[ROADMAP.md](ROADMAP.md)
- **设计蓝图**：[blueprints/README.md](blueprints/README.md)

### Phase 1 - 基础功能验证（当前阶段）
- **目标**：2周内验证核心识别功能
- **技术**：SigLIP (多语言) + BLIP (图像理解) + PaddleOCR + SQLite
- **文档**：[blueprints/phase1/README.md](blueprints/phase1/README.md)

### Phase 2 - 语义搜索增强（下一阶段）
- **目标**：1个月实现智能语义搜索  
- **技术**：SigLIP + BLIP + GroundingDINO (可选) + 混合搜索
- **状态**：待Phase 1验证后启动

### Phase Final - 生产级系统（最终目标）
- **目标**：完整的AI图片管理平台
- **技术**：PostgreSQL + pgvector + Celery + Redis
- **文档**：[blueprints/phase_final/README.md](blueprints/phase_final/README.md)
- **决策**：[decisions/TECHNICAL_DECISIONS.md](decisions/TECHNICAL_DECISIONS.md)

## 📦 依赖版本

查看 [DEPENDENCIES.md](DEPENDENCIES.md) 了解所有依赖的最新版本（2024年11月）

## 🚀 快速开始

### 环境准备（Python 3.12 + uv）
```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或 brew install uv (macOS)

# 验证 Python 版本
python --version  # 应该是 3.12.x
```

### 选项1：运行 Phase 1（推荐先验证）
```bash
cd blueprints/phase1
uv venv --python 3.12
source .venv/bin/activate
uv pip sync requirements.txt
uv run python download_models.py  # 首次运行，预下载模型
uv run python process_dataset.py
```

### 选项2：查看 Phase Final 设计
```bash
cd blueprints/phase_final
# 查看设计文档
cat README.md
# 查看技术决策
cat ../../decisions/TECHNICAL_DECISIONS.md
```

## 📊 技术选型

- **图像理解**：SigLIP (google/siglip-base-patch16-224-i18n) + BLIP (Salesforce/blip-image-captioning-base)
- **物体检测**：GroundingDINO (Phase 2可选增强)
- **OCR**：PaddleOCR 3.3.1
- **深度学习**：PyTorch 2.9.0 + Transformers 4.57.1
- **向量存储**：PostgreSQL + pgvector（主方案）
- **任务队列**：Celery + Redis
- **Web框架**：FastAPI 0.121.1
- **UI**：Streamlit 1.51.0 (Phase 1) / Gradio 5.49.1 (Phase Final)

## 📝 开发状态

- ✅ 产品路线图制定完成
- ✅ 术语统一：Phase 1/2/Final
- ✅ 文档重组完成：blueprints/目录
- ✅ 技术决策明确：PostgreSQL + pgvector
- ✅ Python 3.12 固定
- 🚧 Phase 1 实施准备就绪
- ⏳ Phase 2 待 Phase 1 验证后启动
- ⏳ Phase Final 架构设计完成

## 📄 许可证

[MIT License](LICENSE)