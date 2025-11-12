#!/usr/bin/env python3
"""
Phase 1 数据集处理脚本
用于处理测试数据集，支持增量处理

使用方法:
1. 安装 uv: curl -LsSf https://astral.sh/uv/install.sh | sh
2. 安装依赖: uv pip sync requirements.txt
3. 运行脚本: uv run python process_dataset.py

注意: 必须使用 uv 运行，不要直接使用 python
"""

import asyncio
import logging
from pathlib import Path
import yaml
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logging(config: dict) -> logging.Logger:
    """配置日志系统"""
    log_config = config.get('logging', {})
    
    # 创建日志目录
    log_dir = Path(log_config.get('directory', 'log'))
    log_dir.mkdir(exist_ok=True)
    
    # 日志文件路径
    log_file = log_dir / 'process_dataset.log'
    
    # 配置日志格式
    log_format = log_config.get('format', '[%(asctime)s] %(levelname)s - %(message)s')
    log_level = log_config.get('level', 'INFO')
    
    # 创建日志处理器
    handlers = [
        logging.StreamHandler(),  # 控制台输出
        RotatingFileHandler(
            log_file,
            maxBytes=log_config.get('rotation', {}).get('max_bytes', 10485760),
            backupCount=log_config.get('rotation', {}).get('backup_count', 5)
        )  # 文件输出（带轮转）
    ]
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path) as f:
        return yaml.safe_load(f)


async def main():
    """主处理函数"""
    # 1. 加载配置
    config = load_config()
    
    # 2. 设置日志
    logger = setup_logging(config)
    logger.info("配置加载成功")
    
    # 3. 检查数据集目录
    dataset_dir = Path(config['dataset']['directory'])
    if not dataset_dir.exists():
        logger.error(f"数据集目录不存在: {dataset_dir}")
        logger.info(f"请将测试图片放入 '{dataset_dir}' 目录")
        return
    
    # 统计图片数量
    image_count = 0
    for fmt in config['dataset']['supported_formats']:
        image_count += len(list(dataset_dir.glob(f"**/*{fmt}")))
        image_count += len(list(dataset_dir.glob(f"**/*{fmt.upper()}")))
    
    logger.info(f"数据集目录: {dataset_dir}")
    logger.info(f"发现图片数量: {image_count}")
    
    if image_count == 0:
        logger.warning("未找到任何图片文件")
        return
    
    # 3. 初始化组件
    from processors.preprocessor import ImagePreprocessor
    from processors.batch import BatchProcessor
    from processors.detector import SigLIPBLIPDetector
    from processors.ocr import PaddleOCREngine
    from app.database import get_db_session
    
    # 创建必要的目录
    # 只创建实际需要的缓存目录
    for path_key in ['processed', 'thumbnails', 'embeddings', 'detections', 'ocr_results']:
        Path(config['preprocessing']['paths'][path_key]).mkdir(parents=True, exist_ok=True)
    
    # 确保数据库和状态文件的父目录存在
    Path(config['preprocessing']['paths']['database']).parent.mkdir(parents=True, exist_ok=True)
    Path(config['preprocessing']['paths']['state']).parent.mkdir(parents=True, exist_ok=True)
    
    # 初始化处理器
    preprocessor = ImagePreprocessor(config['preprocessing'])
    detector = SigLIPBLIPDetector(
        model=config['detection']['model'],
        device=config['detection']['device']
    )
    ocr_engine = PaddleOCREngine(config['ocr']) if config['ocr']['enabled'] else None
    
    # 获取数据库会话
    db_session = get_db_session()
    
    # 创建批处理器
    batch_processor = BatchProcessor(
        db_session=db_session,
        detector=detector,
        ocr_engine=ocr_engine,
        preprocessor=preprocessor,
        config=config
    )
    
    # 4. 执行处理
    start_time = datetime.now()
    logger.info("开始处理数据集...")
    
    try:
        # 执行增量处理
        await batch_processor.process_dataset(
            incremental=config['dataset']['incremental']
        )
        
        # 计算处理时间
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"处理完成！总耗时: {elapsed:.1f}秒")
        
        # 显示统计信息
        stats = batch_processor.get_statistics()
        logger.info("处理统计:")
        logger.info(f"  - 总处理: {stats['total_processed']}")
        logger.info(f"  - 成功: {stats['successful']}")
        logger.info(f"  - 跳过(重复): {stats['duplicates']}")
        logger.info(f"  - 失败: {stats['failed']}")
        
        if stats['failed'] > 0:
            logger.warning(f"有 {stats['failed']} 张图片处理失败，请查看日志")
    
    except Exception as e:
        logger.error(f"处理过程中出错: {e}", exc_info=True)
        return 1
    
    # 5. 清理
    logger.info("清理资源...")
    db_session.close()
    
    logger.info("✅ 数据集处理完成！")
    logger.info("下一步：")
    logger.info("  1. 启动API服务: uvicorn app.main:app --reload")
    logger.info("  2. 启动Web界面: streamlit run ui/app.py")
    logger.info("  3. 访问 http://localhost:8501 开始搜索")
    
    return 0


if __name__ == "__main__":
    # 创建日志目录
    Path("logs").mkdir(exist_ok=True)
    
    # 运行主函数
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
