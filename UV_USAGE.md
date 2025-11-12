# 📦 UV Python 包管理器使用规范

## ⚠️ 重要提醒

**本项目强制要求使用 `uv` 作为唯一的Python环境和依赖管理工具**

## 🚫 禁止使用的工具

- ❌ pip
- ❌ pip-tools
- ❌ poetry
- ❌ pipenv
- ❌ conda/mamba
- ❌ virtualenv/venv

## ✅ 必须使用 uv

### 安装 uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 Homebrew (macOS)
brew install uv

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 基本操作

#### 1. 创建虚拟环境

```bash
uv venv
# 会创建 .venv 目录
```

#### 2. 激活虚拟环境

```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

#### 3. 安装依赖

```bash
# 从 requirements.txt 安装
uv pip sync requirements.txt

# 添加新包
uv add package_name

# 添加开发依赖
uv add --dev pytest black

# 指定版本
uv add "fastapi==0.121.1"
```

#### 4. 更新依赖

```bash
# 更新单个包
uv add --upgrade package_name

# 更新所有包
uv pip compile requirements.txt -o requirements.txt --upgrade
```

#### 5. 删除依赖

```bash
uv remove package_name
```

#### 6. 运行脚本

```bash
# 使用 uv run 确保在正确的环境中运行
uv run python script.py
uv run pytest
uv run uvicorn app:main --reload
```

### 项目使用示例

```bash
# 1. 克隆项目
git clone <repo>
cd vibe_photos_v3

# 2. 安装 uv（如果未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 创建虚拟环境
uv venv

# 4. 激活虚拟环境
source .venv/bin/activate

# 5. 安装项目依赖
uv pip sync poc1_design/requirements.txt

# 6. 运行数据处理脚本
uv run python poc1_design/process_dataset.py

# 7. 启动服务
uv run uvicorn app.main:app --reload
uv run streamlit run ui/app.py
```

## 📝 requirements.txt 管理

### 生成 requirements.txt

```bash
# 导出当前环境的依赖
uv pip freeze > requirements.txt

# 或者使用 uv pip compile
uv pip compile requirements.in -o requirements.txt
```

### requirements.in 格式

创建 `requirements.in` 文件，只列出直接依赖：

```txt
fastapi==0.121.1
streamlit==1.51.0
torch==2.9.0
paddlepaddle==2.5.1
```

然后生成完整的 requirements.txt：

```bash
uv pip compile requirements.in -o requirements.txt
```

## 🔍 常见问题

### Q: 为什么必须使用 uv？

A: uv 提供了：
- **极快的速度**：比 pip 快 10-100 倍
- **统一的工具**：同时管理 Python、虚拟环境和依赖
- **锁文件支持**：确保团队环境一致性
- **内存效率**：处理大型依赖树时占用更少内存

### Q: 如何迁移现有的 pip 项目？

```bash
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 创建新的虚拟环境
uv venv

# 3. 激活环境
source .venv/bin/activate

# 4. 同步现有的 requirements.txt
uv pip sync requirements.txt
```

### Q: 如何处理私有包？

```bash
# 使用 index-url
uv add package_name --index-url https://your-private-pypi.com

# 或在 requirements.txt 中指定
--index-url https://pypi.org/simple
--extra-index-url https://your-private-pypi.com
package_name==1.0.0
```

## 📚 参考资源

- [uv 官方文档](https://github.com/astral-sh/uv)
- [uv vs pip 性能对比](https://astral.sh/blog/uv)
- [Python 包管理最佳实践](https://packaging.python.org/)

---

**记住：在本项目中，任何时候都使用 `uv`，不要使用其他Python包管理工具！**
