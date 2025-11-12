# Vibe Photos V3 - 项目根目录

## 📁 项目结构

```
vibe_photos_v3/
├── ROADMAP.md            # 🚀 完整产品路线图
├── FINAL_CHECKLIST.md    # ✅ 最终文档检查清单
├── UV_USAGE.md           # 📦 Python环境管理规范（必读）
├── DIRECTORY_STRUCTURE.md # 📂 目录使用说明
├── poc1_design/          # PoC1设计文档（离线批处理验证）
├── v3_design/            # V3完整设计文档
├── v3_design_feedback/   # 设计审查反馈（含Gemini建议）
├── samples/              # 原始测试数据集（只读）
├── data/                 # 处理结果存储（读写）
├── cache/                # 可复用缓存（跨版本共享）
├── DEPENDENCIES.md       # 所有依赖版本清单
└── LICENSE              # 项目许可证
```

详细目录说明请查看 [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md)

## 🎯 快速导航

### ⚠️ Python环境管理（必读）
- **强制要求**：统一使用 `uv` 管理Python环境
- **禁止使用**：pip/poetry/conda等其他工具
- **使用指南**：[UV_USAGE.md](UV_USAGE.md)

### 📋 产品路线图（NEW）
- **完整规划**：PoC1 → PoC2 → Production
- **时间线**：3-6个月渐进式升级
- **文档**：[ROADMAP.md](ROADMAP.md)

### PoC1 - 基础功能验证（当前阶段）
- **目标**：2周内验证核心识别功能
- **技术**：RTMDet (52.8% mAP) + PaddleOCR + SQLite
- **文档**：[poc1_design/README.md](poc1_design/README.md)

### PoC2 - 语义搜索增强（下一阶段）
- **目标**：1个月实现智能语义搜索
- **技术**：RTMDet + SigLIP + 混合搜索
- **状态**：待PoC1验证后启动

### V3设计 - 生产级系统（最终目标）
- **目标**：完整的AI图片管理平台
- **技术**：多模型集成 + Few-shot学习 + 向量搜索
- **文档**：[v3_design/README.md](v3_design/README.md)

## 📦 依赖版本

查看 [DEPENDENCIES.md](DEPENDENCIES.md) 了解所有依赖的最新版本（2024年11月）

## 🚀 快速开始

### 环境准备（必须先安装 uv）
```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或 brew install uv (macOS)
```

### 选项1：运行PoC1（推荐先验证）
```bash
cd poc1_design
uv venv
source .venv/bin/activate
uv pip sync requirements.txt
uv run python quick_start.py
```

### 选项2：查看V3设计
```bash
cd v3_design
# 查看设计文档
cat README.md
```

## 📊 技术选型

- **物体检测**：RTMDet-L (Apache-2.0许可)
- **OCR**：PaddleOCR 3.3.1
- **深度学习**：PyTorch 2.9.0
- **Web框架**：FastAPI 0.121.1
- **UI**：Streamlit 1.51.0 (PoC1) / Gradio 5.49.1 (V3)

## 📝 开发状态

- ✅ 产品路线图制定完成（基于Gemini反馈优化）
- ✅ PoC1设计完成（支持未来扩展）
- ✅ V3设计愿景完成
- ✅ 依赖版本更新至最新
- 🚧 PoC1实施准备就绪
- ⏳ PoC2待PoC1验证后启动
- ⏳ Production视需求而定

## 📄 许可证

[MIT License](LICENSE)
