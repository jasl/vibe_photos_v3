# Vibe Photos V3 - 项目根目录

## 📁 项目结构

```
vibe_photos_v3/
├── poc1_design/          # PoC1设计文档（离线批处理验证）
├── v3_design/            # V3完整设计文档
├── v3_design_feedback/   # 设计审查反馈
├── DEPENDENCIES.md       # 所有依赖版本清单
└── LICENSE              # 项目许可证
```

## 🎯 快速导航

### PoC1 - 快速验证原型
- **目标**：2周内验证核心识别功能
- **技术**：RTMDet (52.8% mAP) + PaddleOCR + SQLite
- **文档**：[poc1_design/README.md](poc1_design/README.md)

### V3设计 - 完整系统
- **目标**：生产级图片管理系统
- **技术**：多模型集成 + Few-shot学习 + 向量搜索
- **文档**：[v3_design/README.md](v3_design/README.md)

## 📦 依赖版本

查看 [DEPENDENCIES.md](DEPENDENCIES.md) 了解所有依赖的最新版本（2024年11月）

## 🚀 快速开始

### 选项1：运行PoC1（推荐先验证）
```bash
cd poc1_design
pip install -r requirements.txt
python quick_start.py
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

- ✅ V3设计完成
- ✅ PoC1设计完成
- ✅ 依赖版本更新至最新
- 🚧 PoC1实施中
- ⏳ V3实施待定

## 📄 许可证

[MIT License](LICENSE)
