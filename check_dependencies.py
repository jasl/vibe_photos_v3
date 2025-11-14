#!/usr/bin/env python3
"""Dependency compatibility checker for Phase 1."""

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(text: str) -> None:
    """Render a section header with colorized formatting."""
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}")


def check_python_version() -> bool:
    """Validate the active Python interpreter."""
    print_header("Python environment")

    py_version = sys.version_info
    py_version_str = f"{py_version.major}.{py_version.minor}.{py_version.patch}"

    print(f"Python version: {py_version_str}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")

    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 12):
        print(f"{RED}❌ Python 3.12 is required.{RESET}")
        return False

    print(f"{GREEN}✅ Python version is compatible.{RESET}")
    return True


def check_package(package_name: str, import_name: str | None = None, min_version: str | None = None) -> bool:
    """Verify that a package is installed and importable."""
    if import_name is None:
        import_name = package_name

    try:
        installed_version = version(package_name)

        try:
            __import__(import_name)
            status = f"{GREEN}✅{RESET}"

            if min_version and installed_version < min_version:
                status = f"{YELLOW}⚠️  (outdated version){RESET}"
        except ImportError as error:
            status = f"{YELLOW}⚠️  (installed but failed to import: {error}){RESET}"

        print(f"  {status} {package_name:20} {installed_version}")
        return True

    except PackageNotFoundError:
        print(f"  {RED}❌{RESET} {package_name:20} not installed")
        return False


def check_core_dependencies() -> bool:
    """Check core runtime dependencies."""
    print_header("Core dependencies")

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


def check_ai_dependencies() -> None:
    """Verify AI and ML tooling."""
    print_header("AI/ML frameworks")

    pytorch_available = False
    try:
        import torch

        pytorch_available = True
        print(f"  {GREEN}✅{RESET} PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            print(f"     └─ CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("     └─ Apple Silicon MPS available")
        else:
            print("     └─ CPU mode only")

    except ImportError:
        print(f"  {RED}❌{RESET} PyTorch is not installed (required)")

    if pytorch_available:
        check_package("torchvision", "torchvision", "0.24.0")

    print("\n  SigLIP + BLIP dependencies:")
    transformers_available = check_package("transformers", "transformers", "4.57.1")
    if transformers_available:
        print(f"     └─ {GREEN}SigLIP + BLIP stack ready{RESET}")
    else:
        print(f"     └─ {YELLOW}Install transformers to enable SigLIP + BLIP{RESET}")


def check_ocr_dependencies() -> None:
    """Validate OCR stack."""
    print_header("OCR engine")

    paddle_available = False
    try:
        import paddle

        paddle_available = True
        print(f"  {GREEN}✅{RESET} PaddlePaddle {paddle.__version__}")

        if paddle.is_compiled_with_cuda():
            print("     └─ CUDA support enabled")
        else:
            print("     └─ CPU mode")

    except ImportError:
        print(f"  {RED}❌{RESET} PaddlePaddle not installed")

    if paddle_available:
        try:
            from paddleocr import PaddleOCR

            print(f"  {GREEN}✅{RESET} PaddleOCR available")
            print("     └─ Initializing OCR engine...")
            PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            print("     └─ OCR engine initialized successfully")
        except Exception as error:  # noqa: BLE001
            print(f"  {YELLOW}⚠️{RESET} PaddleOCR initialization failed: {error}")


def check_optional_dependencies() -> None:
    """Check optional helper packages."""
    print_header("Optional dependencies")

    optional_packages = [
        ("redis", "redis"),
        ("numpy", "numpy"),
        ("httpx", "httpx"),
    ]

    for pkg, import_name in optional_packages:
        check_package(pkg, import_name)


def run_quick_tests() -> None:
    """Run basic smoke tests for critical libraries."""
    print_header("Quick smoke tests")

    try:
        from fastapi import FastAPI

        FastAPI()
        print(f"  {GREEN}✅{RESET} FastAPI app instantiated")
    except Exception as error:  # noqa: BLE001
        print(f"  {RED}❌{RESET} FastAPI smoke test failed: {error}")

    try:
        from sqlalchemy import create_engine

        create_engine("sqlite:///:memory:")
        print(f"  {GREEN}✅{RESET} SQLAlchemy in-memory DB ready")
    except Exception as error:  # noqa: BLE001
        print(f"  {RED}❌{RESET} SQLAlchemy smoke test failed: {error}")

    try:
        from PIL import Image

        Image.new("RGB", (100, 100))
        print(f"  {GREEN}✅{RESET} Pillow image creation succeeded")
    except Exception as error:  # noqa: BLE001
        print(f"  {RED}❌{RESET} Pillow smoke test failed: {error}")


def print_recommendations() -> None:
    """Summarize remediation steps."""
    print_header("Recommendations")

    print(
        f"""
1. Install missing dependencies with:
   {YELLOW}uv sync{RESET}

2. Optional GPU acceleration:
   - NVIDIA GPU (CUDA 13.0): {YELLOW}uv pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130{RESET}
   - Apple Silicon: PyTorch will automatically use MPS acceleration when available

3. PaddlePaddle fallback index (CPU builds):
   {YELLOW}uv pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/{RESET}

4. Developer tooling bundle:
   {YELLOW}uv pip install -r requirements-dev.txt{RESET}
"""
    )


def main() -> int:
    """Program entrypoint."""
    print(f"{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}  Vibe Photos Phase 1 — Dependency compatibility check{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}")

    python_ok = check_python_version()
    if not python_ok:
        print(f"\n{RED}Upgrade Python before continuing.{RESET}")
        return 1

    core_ok = check_core_dependencies()
    check_ai_dependencies()
    check_ocr_dependencies()
    check_optional_dependencies()
    run_quick_tests()
    print_recommendations()

    print_header("Summary")
    if core_ok:
        print(f"{GREEN}Core dependencies are ready — you can start building!{RESET}")
    else:
        print(f"{YELLOW}Some dependencies are missing. Follow the steps above to resolve them.{RESET}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
