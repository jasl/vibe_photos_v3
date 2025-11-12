#!/usr/bin/env python3
"""
依赖兼容性检查脚本
用于验证所有依赖是否正确安装并能正常工作
"""

import sys
import platform
from importlib.metadata import version, PackageNotFoundError

# ANSI颜色代码
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    """打印标题"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

def check_python_version():
    """检查Python版本"""
    print_header("Python环境检查")
    
    py_version = sys.version_info
    py_version_str = f"{py_version.major}.{py_version.minor}.{py_version.patch}"
    
    print(f"Python版本: {py_version_str}")
    print(f"平台: {platform.platform()}")
    print(f"架构: {platform.machine()}")
    
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        print(f"{RED}❌ Python版本需要 >= 3.8{RESET}")
        return False
    else:
        print(f"{GREEN}✅ Python版本符合要求{RESET}")
        return True

def check_package(package_name, import_name=None, min_version=None):
    """检查单个包是否安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        # 获取版本
        installed_version = version(package_name)
        
        # 尝试导入
        try:
            __import__(import_name)
            status = f"{GREEN}✅{RESET}"
            
            # 版本检查
            if min_version and installed_version < min_version:
                status = f"{YELLOW}⚠️  (版本较旧){RESET}"
        except ImportError as e:
            status = f"{YELLOW}⚠️  (已安装但无法导入: {e}){RESET}"
        
        print(f"  {status} {package_name:20} {installed_version}")
        return True
        
    except PackageNotFoundError:
        print(f"  {RED}❌{RESET} {package_name:20} 未安装")
        return False

def check_core_dependencies():
    """检查核心依赖"""
    print_header("核心依赖检查")
    
    packages = [
        ("fastapi", "fastapi", "0.121.1"),
        ("uvicorn", "uvicorn", "0.38.0"),
        ("streamlit", "streamlit", "1.51.0"),
        ("sqlalchemy", "sqlalchemy", "2.0.44"),
        ("pillow", "PIL", "11.3.0"),
        ("python-multipart", "multipart", "0.0.20"),
        ("aiofiles", "aiofiles", "24.1.0"),
        ("pydantic", "pydantic", "2.11.10"),
    ]
    
    success = True
    for pkg, import_name, min_ver in packages:
        if not check_package(pkg, import_name, min_ver):
            success = False
    
    return success

def check_ai_dependencies():
    """检查AI/ML依赖"""
    print_header("AI/ML框架检查")
    
    # 检查PyTorch
    pytorch_available = False
    try:
        import torch
        pytorch_available = True
        print(f"  {GREEN}✅{RESET} PyTorch {torch.__version__}")
        
        # 检查CUDA/MPS支持
        if torch.cuda.is_available():
            print(f"     └─ CUDA可用: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"     └─ Apple Silicon MPS可用")
        else:
            print(f"     └─ 仅CPU模式")
            
    except ImportError:
        print(f"  {RED}❌{RESET} PyTorch 未安装（必需）")
    
    # 检查TorchVision
    if pytorch_available:
        check_package("torchvision", "torchvision", "0.24.0")
    
    # 检查主要方案：SigLIP+BLIP
    print(f"  \n  SigLIP+BLIP依赖（主要方案）:")
    transformers_available = check_package("transformers", "transformers", "4.57.1")
    if transformers_available:
        print(f"     └─ {GREEN}SigLIP+BLIP方案可用{RESET}")
    else:
        print(f"     └─ {YELLOW}请安装transformers以使用SigLIP+BLIP{RESET}")

def check_ocr_dependencies():
    """检查OCR依赖"""
    print_header("OCR引擎检查")
    
    # 检查PaddlePaddle
    paddle_available = False
    try:
        import paddle
        paddle_available = True
        print(f"  {GREEN}✅{RESET} PaddlePaddle {paddle.__version__}")
        
        # 检查设备支持
        if paddle.is_compiled_with_cuda():
            print(f"     └─ CUDA支持已启用")
        else:
            print(f"     └─ CPU模式")
            
    except ImportError:
        print(f"  {RED}❌{RESET} PaddlePaddle 未安装")
    
    # 检查PaddleOCR
    if paddle_available:
        try:
            from paddleocr import PaddleOCR
            print(f"  {GREEN}✅{RESET} PaddleOCR 已安装")
            
            # 测试OCR功能
            print(f"     └─ 正在测试OCR功能...")
            ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
            print(f"     └─ OCR引擎初始化成功")
        except Exception as e:
            print(f"  {YELLOW}⚠️{RESET}  PaddleOCR 安装但初始化失败: {e}")

def check_optional_dependencies():
    """检查可选依赖"""
    print_header("可选依赖检查")
    
    optional_packages = [
        ("redis", "redis"),
        ("numpy", "numpy"),
        ("httpx", "httpx"),
    ]
    
    for pkg, import_name in optional_packages:
        check_package(pkg, import_name)

def run_quick_tests():
    """运行快速功能测试"""
    print_header("快速功能测试")
    
    # 测试FastAPI
    try:
        from fastapi import FastAPI
        app = FastAPI()
        print(f"  {GREEN}✅{RESET} FastAPI应用创建成功")
    except Exception as e:
        print(f"  {RED}❌{RESET} FastAPI测试失败: {e}")
    
    # 测试SQLAlchemy
    try:
        from sqlalchemy import create_engine
        engine = create_engine("sqlite:///:memory:")
        print(f"  {GREEN}✅{RESET} SQLAlchemy数据库连接成功")
    except Exception as e:
        print(f"  {RED}❌{RESET} SQLAlchemy测试失败: {e}")
    
    # 测试Pillow
    try:
        from PIL import Image
        img = Image.new('RGB', (100, 100))
        print(f"  {GREEN}✅{RESET} Pillow图像处理可用")
    except Exception as e:
        print(f"  {RED}❌{RESET} Pillow测试失败: {e}")

def print_recommendations():
    """打印建议"""
    print_header("建议和下一步")
    
    print(f"""
1. 如有缺失的依赖，请运行：
   {YELLOW}uv pip sync requirements.txt{RESET}

2. 对于GPU加速（可选）：
   - NVIDIA GPU (CUDA 12.4): {YELLOW}uv pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu124{RESET}
   - NVIDIA GPU (CUDA 12.1): {YELLOW}uv pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu121{RESET}
   - Apple Silicon: PyTorch会自动使用MPS加速

3. 如遇到PaddlePaddle安装问题：
   {YELLOW}uv pip install paddlepaddle==3.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple{RESET}

4. 开发工具（可选）：
   {YELLOW}uv pip sync requirements.txt requirements-dev.txt{RESET}
""")

def main():
    """主函数"""
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}  Vibe Photos Phase 1 - 依赖兼容性检查{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    # 运行检查
    python_ok = check_python_version()
    if not python_ok:
        print(f"\n{RED}请先升级Python版本{RESET}")
        return 1
    
    core_ok = check_core_dependencies()
    check_ai_dependencies()
    check_ocr_dependencies()
    check_optional_dependencies()
    run_quick_tests()
    print_recommendations()
    
    # 总结
    print_header("检查完成")
    if core_ok:
        print(f"{GREEN}核心依赖已就绪，可以开始开发！{RESET}")
    else:
        print(f"{YELLOW}部分依赖缺失，请根据上述建议安装{RESET}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
