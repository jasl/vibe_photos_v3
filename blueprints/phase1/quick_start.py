#!/usr/bin/env python3
"""
Phase 1 快速启动脚本 - 用于快速搭建和验证环境
"""

import os
import sys
import subprocess
from pathlib import Path
import json

def print_step(step_num, total, message):
    """打印步骤信息"""
    print(f"\n[{step_num}/{total}] {message}")
    print("-" * 50)

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python版本需要 >= 3.8")
        print(f"   当前版本: {sys.version}")
        return False
    print(f"✅ Python版本: {sys.version.split()[0]}")
    return True

def create_project_structure():
    """创建项目目录结构"""
    directories = [
        "phase1/app",
        "phase1/app/api",
        "phase1/processors",
        "phase1/ui",
        "phase1/scripts",
        "phase1/tests",
        "phase1/data/images",
        "phase1/data/thumbnails",
        "phase1/data/cache",
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    print(f"✅ 创建了 {len(directories)} 个目录")
    return True

def create_requirements_file():
    """创建requirements.txt文件"""
    requirements = """# 核心依赖 - 2024年11月最新稳定版本
fastapi==0.121.1
uvicorn==0.38.0
streamlit==1.51.0
sqlalchemy==2.0.44
pillow==11.3.0
python-multipart==0.0.20
aiofiles==24.1.0
pydantic==2.11.10

# 识别引擎（推荐SigLIP+BLIP）
# Option A: 使用SigLIP+BLIP（推荐，多语言支持，~85%准确率）
torch==2.9.0
torchvision==0.24.0
transformers==4.57.1
sentence-transformers==5.1.2

# Option B: 使用SigLIP（备选，更强大）
# transformers==4.57.1
# 模型: google/siglip-base-patch16-224-i18n

# OCR引擎
paddlepaddle==3.2.0
paddleocr==3.3.1

# 开发工具
pytest==9.0.0
black==25.11.0
ruff==0.14.4
"""
    
    with open("phase1/requirements.txt", "w") as f:
        f.write(requirements)
    
    print("✅ 创建 requirements.txt")
    return True

def create_config_file():
    """创建配置文件"""
    config = {
        "database": {
            "url": "sqlite:///./phase1.db"
        },
        "batch": {
            "size": 10,
            "max_workers": 4
        },
        "image": {
            "thumbnail_size": [256, 256],
            "supported_formats": [".jpg", ".jpeg", ".png", ".webp"]
        },
        "detection": {
            "model": "siglip-base",  # 使用SigLIP模型
            "confidence_threshold": 0.3,
            "device": "cpu"  # 或 "cuda"
        },
        "ocr": {
            "languages": ["ch", "en"],
            "enable": True
        },
        "search": {
            "limit": 50
        },
        "paths": {
            "upload": "data/images",
            "thumbnails": "data/thumbnails",
            "cache": "data/cache"
        }
    }
    
    with open("phase1/config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("✅ 创建 config.json")
    return True

def create_sample_scripts():
    """创建示例脚本"""
    
    # 1. 数据库初始化脚本
    init_db_script = '''"""数据库初始化脚本"""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Text

Base = declarative_base()

class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    
print("✅ 数据库模型已定义")
'''
    
    # 2. 简单的主应用
    main_app_script = '''"""FastAPI主应用"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Vibe Photos Phase 1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Vibe Photos Phase 1 API", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
'''
    
    # 3. Streamlit UI
    ui_script = '''"""Streamlit UI应用"""
import streamlit as st

st.set_page_config(
    page_title="Vibe Photos Phase 1",
    page_icon="📸",
    layout="wide"
)

st.title("📸 Vibe Photos Phase 1")

# 侧边栏
with st.sidebar:
    st.header("功能选择")
    page = st.radio(
        "选择功能",
        ["批量导入", "搜索浏览", "处理状态"]
    )

# 主页面
if page == "批量导入":
    st.header("批量导入图片")
    folder_path = st.text_input("图片文件夹路径")
    if st.button("开始导入"):
        st.success("导入功能开发中...")
        
elif page == "搜索浏览":
    st.header("搜索和浏览")
    search_query = st.text_input("搜索关键词")
    if search_query:
        st.info(f"搜索: {search_query}")
        
elif page == "处理状态":
    st.header("处理状态")
    st.info("状态监控功能开发中...")
'''
    
    # 保存脚本
    with open("phase1/scripts/init_db.py", "w") as f:
        f.write(init_db_script)
    
    with open("phase1/app/main.py", "w") as f:
        f.write(main_app_script)
        
    with open("phase1/ui/app.py", "w") as f:
        f.write(ui_script)
    
    # 创建__init__.py文件
    for path in ["phase1/app/__init__.py", "phase1/processors/__init__.py"]:
        Path(path).touch()
    
    print("✅ 创建示例脚本")
    return True

def create_readme():
    """创建README文件"""
    readme = """# Vibe Photos Phase 1

## 快速开始

### 1. 安装依赖
```bash
cd phase1
pip install -r requirements.txt
```

### 2. 初始化数据库
```bash
python scripts/init_db.py
```

### 3. 启动服务

#### 启动API服务
```bash
uvicorn app.main:app --reload --port 8000
```

#### 启动Web UI（新终端）
```bash
streamlit run ui/app.py --server.port 8501
```

### 4. 访问服务
- API文档: http://localhost:8000/docs
- Web界面: http://localhost:8501

## 项目结构
```
phase1/
├── app/           # FastAPI应用
├── processors/    # 处理引擎
├── ui/           # Streamlit界面
├── scripts/      # 工具脚本
├── tests/        # 测试文件
├── data/         # 数据目录
└── config.json   # 配置文件
```

## 下一步
1. 完善批处理功能
2. 集成识别引擎
3. 实现搜索功能
"""
    
    with open("phase1/README.md", "w") as f:
        f.write(readme)
    
    print("✅ 创建 README.md")
    return True

def main():
    """主函数"""
    print("=" * 50)
    print("   Vibe Photos Phase 1 - 快速启动脚本")
    print("=" * 50)
    
    total_steps = 6
    
    # Step 1: 检查Python版本
    print_step(1, total_steps, "检查Python版本")
    if not check_python_version():
        print("\n❌ 请先升级Python版本")
        return 1
    
    # Step 2: 创建项目结构
    print_step(2, total_steps, "创建项目结构")
    if not create_project_structure():
        return 1
    
    # Step 3: 创建requirements.txt
    print_step(3, total_steps, "创建依赖文件")
    if not create_requirements_file():
        return 1
    
    # Step 4: 创建配置文件
    print_step(4, total_steps, "创建配置文件")
    if not create_config_file():
        return 1
    
    # Step 5: 创建示例脚本
    print_step(5, total_steps, "创建示例脚本")
    if not create_sample_scripts():
        return 1
    
    # Step 6: 创建README
    print_step(6, total_steps, "创建文档")
    if not create_readme():
        return 1
    
    # 完成
    print("\n" + "=" * 50)
    print("✅ Phase 1环境搭建完成！")
    print("=" * 50)
    print("\n下一步操作：")
    print("1. cd phase1")
    print("2. pip install -r requirements.txt")
    print("3. uvicorn app.main:app --reload")
    print("4. 访问 http://localhost:8000/docs")
    print("\n祝开发顺利！🚀")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
