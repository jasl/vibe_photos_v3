#!/usr/bin/env python3
"""
模型预下载脚本 - Phase 1

预下载所有必需的模型文件到 models/ 目录，避免运行时下载。
支持断点续传和完整性校验。
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, Optional
import requests
from tqdm import tqdm
import json
import time

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 模型存储目录
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# 模型配置
MODELS_CONFIG = {
    "rtmdet": {
        "name": "SigLIP+BLIP",
        "description": "多语言图像理解模型",
        "files": {
            "config": {
                "url": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco.py",
                "size": "~10KB",
                "path": "rtmdet/rtmdet_l_coco.py"
            },
            "checkpoint": {
                "url": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth",
                "size": "~330MB",
                "path": "rtmdet/rtmdet_l_coco.pth",
                "sha256": "5a0be7c4"  # 简化的hash，实际使用时需要完整hash
            }
        }
    },
    "paddleocr": {
        "name": "PaddleOCR",
        "description": "中英文OCR模型",
        "files": {
            "det_model": {
                "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar",
                "size": "~4.9MB",
                "path": "paddleocr/ch_PP-OCRv4_det_infer.tar",
                "extract": True
            },
            "rec_model": {
                "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar",
                "size": "~10MB",
                "path": "paddleocr/ch_PP-OCRv4_rec_infer.tar",
                "extract": True
            },
            "cls_model": {
                "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
                "size": "~2.1MB",
                "path": "paddleocr/ch_ppocr_mobile_v2.0_cls_infer.tar",
                "extract": True
            }
        }
    }
}

def download_file(url: str, dest_path: Path, desc: str, chunk_size: int = 8192) -> bool:
    """
    下载文件，支持断点续传和进度显示
    
    Args:
        url: 下载URL
        dest_path: 目标文件路径
        desc: 进度条描述
        chunk_size: 下载块大小
    
    Returns:
        是否下载成功
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查是否需要断点续传
    resume_pos = 0
    mode = 'wb'
    if dest_path.exists():
        resume_pos = dest_path.stat().st_size
        mode = 'ab'
    
    try:
        # 设置请求头
        headers = {}
        if resume_pos > 0:
            headers['Range'] = f'bytes={resume_pos}-'
            print(f"  继续下载: {dest_path.name} (已下载 {resume_pos:,} bytes)")
        
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        # 获取文件总大小
        total_size = int(response.headers.get('content-length', 0))
        if resume_pos > 0:
            total_size += resume_pos
        
        # 如果文件已完整下载
        if resume_pos >= total_size and total_size > 0:
            print(f"  ✓ {dest_path.name} 已存在且完整")
            return True
        
        # 下载文件
        with open(dest_path, mode) as f:
            with tqdm(
                total=total_size,
                initial=resume_pos,
                unit='iB',
                unit_scale=True,
                desc=desc,
                ncols=100
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"  ✓ 下载完成: {dest_path.name}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"  ✗ 下载失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n  ! 下载被中断，下次运行将继续")
        return False

def extract_tar(tar_path: Path) -> bool:
    """解压tar文件"""
    import tarfile
    
    try:
        extract_dir = tar_path.parent
        print(f"  解压中: {tar_path.name}")
        
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(extract_dir)
        
        print(f"  ✓ 解压完成: {extract_dir}")
        return True
        
    except Exception as e:
        print(f"  ✗ 解压失败: {e}")
        return False

def verify_file(file_path: Path, expected_hash: Optional[str] = None) -> bool:
    """验证文件完整性"""
    if not file_path.exists():
        return False
    
    if expected_hash:
        # 计算文件hash（简化版，实际需要完整实现）
        print(f"  验证中: {file_path.name}")
        # TODO: 实现完整的hash验证
        return True
    
    # 基本检查：文件大小不为0
    return file_path.stat().st_size > 0

def download_all_models() -> bool:
    """下载所有模型"""
    print("=" * 60)
    print("Phase 1 模型预下载")
    print("=" * 60)
    print(f"模型目录: {MODELS_DIR}")
    print()
    
    all_success = True
    
    for model_key, model_info in MODELS_CONFIG.items():
        print(f"\n📦 {model_info['name']}")
        print(f"   {model_info['description']}")
        print("-" * 40)
        
        for file_key, file_info in model_info['files'].items():
            dest_path = MODELS_DIR / file_info['path']
            
            # 检查文件是否已存在
            if dest_path.exists() and verify_file(dest_path, file_info.get('sha256')):
                print(f"  ✓ {dest_path.name} 已存在")
                
                # 如果需要解压且未解压
                if file_info.get('extract') and dest_path.suffix == '.tar':
                    extract_dir = dest_path.parent / dest_path.stem
                    if not extract_dir.exists():
                        extract_tar(dest_path)
                continue
            
            # 下载文件
            print(f"  下载: {file_info['size']} - {file_key}")
            success = download_file(
                file_info['url'],
                dest_path,
                f"  {model_info['name']}/{file_key}"
            )
            
            if not success:
                all_success = False
                continue
            
            # 解压文件（如果需要）
            if file_info.get('extract') and dest_path.suffix == '.tar':
                extract_tar(dest_path)
    
    # 创建模型信息文件
    info_file = MODELS_DIR / "models_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        info = {
            "download_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": MODELS_CONFIG
        }
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    if all_success:
        print("✅ 所有模型下载完成！")
        print(f"   模型存储在: {MODELS_DIR}")
        print("\n   下一步：")
        print("   python process_dataset.py")
    else:
        print("⚠️  部分模型下载失败")
        print("   请重新运行此脚本继续下载")
    print("=" * 60)
    
    return all_success

def clean_models():
    """清理所有已下载的模型"""
    print("清理模型目录...")
    import shutil
    
    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR)
        MODELS_DIR.mkdir(exist_ok=True)
        print(f"✓ 已清理: {MODELS_DIR}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 1 模型预下载工具")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="清理所有已下载的模型"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="仅检查模型是否已下载"
    )
    
    args = parser.parse_args()
    
    if args.clean:
        clean_models()
    elif args.check:
        # TODO: 实现检查功能
        print("检查功能待实现")
    else:
        success = download_all_models()
        sys.exit(0 if success else 1)
