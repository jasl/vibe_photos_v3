# Phase 1: 离线批处理 + 快速检索验证系统

## 📊 测试数据集

- **规模**：30,112张PNG图片
- **总大小**：400GB
- **时间跨度**：2012-2025年（14年）
- **地理分布**：中国、美国、日本、新加坡等多个国家
- **特点**：真实个人照片库，符合目标用户场景

## 🎯 目标与定位

### 核心目标
验证Phase Final设计中的核心识别功能质量，采用**最简单可行**的架构，专注于：
1. **验证识别准确性** - 不追求性能，先证明识别能力
2. **降低系统复杂度** - 避免过早优化
3. **快速验证反馈** - 2周内可交付可测试的原型

### Phase 1阶段原则
- ✅ **可破坏性改动** - 不考虑向后兼容
- ✅ **无数据迁移** - 每次运行可以重新开始
- ✅ **快速试错** - 失败了就推倒重来
- ✅ **灵活调整** - 根据效果随时改变方向

### 范围界定
**包含：**
- ✅ 离线批量图片导入和处理
- ✅ 多语言图像分类（使用SigLIP，~85%准确率）
- ✅ 图像理解和描述（使用BLIP生成自然语言描述）
- ✅ OCR文本提取功能（PaddleOCR）
- ✅ 简单的元数据搜索
- ✅ 基础Web UI展示和搜索

**不包含：**
- ❌ 实时处理和性能优化
- ❌ Few-shot学习功能
- ❌ 品牌/型号细分识别
- ❌ 向量搜索和复杂检索
- ❌ 用户认证和多租户

## 📊 成功标准

1. **功能验证**
   - 能够批量处理30,000+张图片（实际测试数据集）
   - 大类识别准确率达到70-80%
   - OCR提取准确率达到85%以上
   - 搜索功能可用且响应合理

2. **用户体验**
   - 批处理完成后能快速浏览结果
   - 搜索结果相关性高
   - UI简洁直观，无需培训即可使用

3. **技术验证**
   - 架构简单，易于理解和修改
   - 依赖最少，部署简单
   - 代码清晰，易于后续迭代

## 🏗 简化架构

```
用户 → Web UI → FastAPI → 批处理器 → 数据库
                  ↓          ↓
              搜索API    识别引擎
                         (SigLIP+BLIP + OCR)
```

## 📅 时间规划

- **第1-3天**: 搭建基础框架，实现批处理流程
- **第4-7天**: 集成识别引擎，完成核心功能
- **第8-10天**: 开发简单UI，实现搜索功能
- **第11-14天**: 测试优化，准备演示

## 📚 文档导航

### 核心文档
1. [技术架构](architecture.md) - 系统架构和技术栈
2. [实施计划](implementation.md) - 开发步骤和代码示例
3. [测试方案](testing.md) - 测试策略和验证方法
4. [设计决策](design_decisions.md) - 关键决策和权衡

### 使用指南
5. **[数据集使用](DATASET_USAGE.md)** - 测试数据集处理指南 ⭐
6. **[配置文件](config.yaml)** - 系统配置示例 ⭐
7. **[处理脚本](process_dataset.py)** - 数据集批处理脚本 ⭐
8. **[模型下载](download_models.py)** - 预训练模型下载工具 ⭐ NEW

## 🚀 快速开始

### ⚠️ 重要：统一使用 `uv` 管理Python环境

**本项目必须使用 `uv` 管理所有Python依赖，请勿使用 pip/venv/poetry**

```bash
# 安装 uv (如果未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或使用 Homebrew (macOS)
brew install uv

# 1. 准备测试数据集
mkdir -p samples
# 将测试图片放入samples目录

# 2. 使用 uv 创建虚拟环境和安装依赖
uv venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# 3. 使用 uv 同步依赖
uv pip sync requirements.txt
# 或者使用 uv add 安装新包
# uv add fastapi streamlit

# 4. 预下载模型文件（首次运行，约430MB）
uv run python download_models.py
# 检查模型: uv run python download_models.py --check
# 清理模型: uv run python download_models.py --clean

# 5. 处理数据集（支持增量处理）
uv run python process_dataset.py

# 6. 启动服务
uv run uvicorn app.main:app --reload
uv run streamlit run ui/app.py
```
