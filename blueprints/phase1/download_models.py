#!/usr/bin/env python
"""
模型下载工具 - Vibe Photos Phase 1

用于下载和准备所需的预训练模型。
SigLIP和BLIP模型会在首次使用时通过transformers自动下载。
"""

import os
import sys
import time
import hashlib
import tarfile
import zipfile
import requests
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse


# 配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache"

# 创建必要的目录
MODELS_DIR.mkdir(exist_ok=True)

# 模型配置（注意：SigLIP和BLIP会通过transformers自动下载）
MODELS_CONFIG = {
    "models_info": {
        "name": "Vibe Photos Phase 1 模型集",
        "description": "包含SigLIP、BLIP和PaddleOCR模型",
        "note": "SigLIP和BLIP模型会在首次运行时自动下载"
    },
    
    "auto_download": {
        "siglip": {
            "name": "google/siglip-base-patch16-224-i18n",
            "description": "多语言图像分类模型",
            "size": "~400MB",
            "source": "Hugging Face",
            "note": "通过transformers库自动下载"
        },
        "blip": {
            "name": "Salesforce/blip-image-captioning-base",
            "description": "图像描述生成模型",
            "size": "~990MB",
            "source": "Hugging Face",
            "note": "通过transformers库自动下载"
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
                "size": "~2.2MB",
                "path": "paddleocr/ch_ppocr_mobile_v2.0_cls_infer.tar",
                "extract": True
            }
        }
    }
}


class ModelDownloader:
    """模型下载器"""
    
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url: str, dest_path: Path, 
                     expected_size: Optional[str] = None) -> bool:
        """下载文件
        
        Args:
            url: 下载URL
            dest_path: 目标路径
            expected_size: 预期文件大小（用于显示）
            
        Returns:
            是否下载成功
        """
        # 如果文件已存在，跳过下载
        if dest_path.exists():
            print(f"✓ 文件已存在: {dest_path.name}")
            return True
            
        # 确保目标目录存在
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"下载: {dest_path.name}")
        if expected_size:
            print(f"  预期大小: {expected_size}")
            
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = downloaded / total_size * 100
                            sys.stdout.write(f"\r  进度: {progress:.1f}%")
                            sys.stdout.flush()
                            
            print(f"\n✓ 下载完成: {dest_path.name}")
            return True
            
        except Exception as e:
            print(f"\n✗ 下载失败: {e}")
            if dest_path.exists():
                dest_path.unlink()
            return False
            
    def extract_archive(self, archive_path: Path) -> bool:
        """解压缩文件
        
        Args:
            archive_path: 压缩文件路径
            
        Returns:
            是否解压成功
        """
        extract_dir = archive_path.parent
        
        try:
            if archive_path.suffix == '.tar':
                with tarfile.open(archive_path, 'r') as tar:
                    tar.extractall(extract_dir)
            elif archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            else:
                print(f"不支持的压缩格式: {archive_path.suffix}")
                return False
                
            print(f"✓ 解压完成: {archive_path.name}")
            return True
            
        except Exception as e:
            print(f"✗ 解压失败: {e}")
            return False
            
    def download_paddleocr_models(self) -> bool:
        """下载PaddleOCR模型"""
        print("\n" + "="*50)
        print("下载PaddleOCR模型")
        print("="*50)
        
        paddleocr_config = MODELS_CONFIG['paddleocr']
        success = True
        
        for file_key, file_info in paddleocr_config['files'].items():
            dest_path = self.models_dir / file_info['path']
            
            if not self.download_file(
                file_info['url'], 
                dest_path, 
                file_info.get('size')
            ):
                success = False
                continue
                
            # 如果需要解压
            if file_info.get('extract'):
                if not self.extract_archive(dest_path):
                    success = False
                    
        return success
        
    def show_auto_download_info(self):
        """显示自动下载模型的信息"""
        print("\n" + "="*50)
        print("自动下载模型说明")
        print("="*50)
        
        auto_models = MODELS_CONFIG['auto_download']
        
        for model_key, model_info in auto_models.items():
            print(f"\n{model_info['name']}")
            print(f"  描述: {model_info['description']}")
            print(f"  大小: {model_info['size']}")
            print(f"  来源: {model_info['source']}")
            print(f"  说明: {model_info['note']}")
            
        print("\n这些模型会在首次运行程序时自动下载到Hugging Face缓存目录。")
        print("缓存位置通常为: ~/.cache/huggingface/hub/")
        
    def download_all(self) -> bool:
        """下载所有模型"""
        print("\n" + "="*50)
        print("Vibe Photos Phase 1 模型下载工具")
        print("="*50)
        
        # 显示自动下载模型信息
        self.show_auto_download_info()
        
        # 下载PaddleOCR模型
        if not self.download_paddleocr_models():
            print("\n⚠️ 部分模型下载失败，请检查网络连接后重试")
            return False
            
        print("\n" + "="*50)
        print("✅ 所有手动下载的模型已准备就绪！")
        print("="*50)
        print("\n提示：")
        print("1. SigLIP和BLIP模型会在首次运行时自动下载")
        print("2. 如需预先下载，可运行以下Python代码：")
        print("\n```python")
        print("from transformers import AutoModel, AutoProcessor, BlipForConditionalGeneration, BlipProcessor")
        print("# 下载SigLIP")
        print("AutoModel.from_pretrained('google/siglip-base-patch16-224-i18n')")
        print("AutoProcessor.from_pretrained('google/siglip-base-patch16-224-i18n')")
        print("# 下载BLIP")
        print("BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')")
        print("BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')")
        print("```")
        
        return True


def main():
    """主函数"""
    downloader = ModelDownloader()
    
    if downloader.download_all():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()