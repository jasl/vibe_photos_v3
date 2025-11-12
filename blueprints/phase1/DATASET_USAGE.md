# Phase 1 测试数据集使用指南

## 📋 概述

Phase 1阶段**不支持图片上传**，所有测试使用本地数据集目录。系统支持增量处理，可以随时添加新图片到数据集中。

## 🚀 快速开始

### 1. 准备测试数据集

```bash
# 创建数据集目录（如果不存在）
mkdir -p samples

# 将测试图片复制到samples目录
# 支持多层子目录结构
samples/
├── November 1, 2025/
│   ├── IMG_001.jpg
│   └── IMG_002.heic
├── Beijing, October 29, 2025/
│   ├── photo1.png
│   └── photo2.jpeg
└── test_images/
    └── sample.webp
```

### 2. 配置文件设置

编辑 `config.yaml`：

```yaml
dataset:
  directory: "samples"  # 数据集目录
  incremental: true     # 启用增量处理
  
  supported_formats:
    - .jpg
    - .jpeg
    - .png
    - .heic
    - .webp
```

### 3. 首次处理数据集

```bash
# 安装 uv（如果未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境（使用uv）
uv venv
source .venv/bin/activate

# 安装依赖（使用uv）
uv pip sync requirements.txt

# 处理测试数据集（使用uv）
uv run python process_dataset.py
```

输出示例：
```
[2024-11-12 10:00:00] INFO - 数据集目录: samples
[2024-11-12 10:00:00] INFO - 发现图片数量: 1234
[2024-11-12 10:00:01] INFO - 开始处理数据集...
[2024-11-12 10:05:00] INFO - 处理完成！总耗时: 300.0秒
[2024-11-12 10:05:00] INFO - 处理统计:
  - 总处理: 1234
  - 成功: 1200
  - 跳过(重复): 30
  - 失败: 4
```

### 4. 增量添加新图片

```bash
# 添加新图片到数据集
cp new_photos/*.jpg samples/

# 重新运行处理脚本（只处理新图片，使用uv）
uv run python process_dataset.py
```

系统会自动：
- 跳过已处理的图片
- 检测并跳过重复图片（基于感知哈希）
- 只处理新添加的图片

## 🔄 图像预处理流程

每张图片都会经过以下处理：

1. **感知哈希计算** - 用于去重检测
2. **格式归一化** - 统一转换为JPEG/RGB
3. **EXIF处理** - 自动旋转到正确方向
4. **尺寸限制** - 最大边长4096px
5. **缩略图生成** - 512x512预览图
6. **图像理解** - SigLIP+BLIP识别
7. **文本提取** - PaddleOCR提取文字
8. **索引更新** - 更新搜索索引

## 📁 目录结构

处理后的文件组织（分离数据和缓存）：

```
samples/                    # 原始数据（只读）
├── November 1, 2025/
├── Beijing, October 29, 2025/
└── ...

data/                      # 数据库和状态（版本特定）
├── vibe_photos.db         # SQLite数据库
└── processing_state.json  # 处理状态

cache/                     # 可复用缓存（跨版本共享）
├── images/
│   ├── processed/        # 归一化后的图片
│   │   ├── a1b2c3d4.jpg
│   │   └── e5f6g7h8.jpg
│   └── thumbnails/       # 缩略图
│       ├── a1b2c3d4_thumb.jpg
│       └── e5f6g7h8_thumb.jpg
├── detections/           # 检测结果缓存
│   ├── a1b2c3d4.json
│   └── e5f6g7h8.json
├── ocr/                  # OCR结果缓存
│   ├── a1b2c3d4.json
│   └── e5f6g7h8.json
└── hashes/               # 哈希缓存
    └── phash_cache.json
```

**注意**: 缓存文件名使用感知哈希（phash）命名，确保内容相同的图片共享缓存。

## ⚙️ 高级配置

### 去重设置

调整感知哈希参数以控制去重灵敏度：

```yaml
preprocessing:
  deduplication:
    enabled: true
    algorithm: "phash"
    hash_size: 8    # 增大提高精度（4-16）
    threshold: 5    # 减小使去重更严格（0-10）
```

### 批处理优化

```yaml
batch_processing:
  batch_size: 10      # 根据内存调整
  max_workers: 8      # 根据CPU核心数调整
```

### 检测过滤

只检测特定类别的物体：

```yaml
detection:
  filter_classes:
    - person
    - laptop
    - cell_phone
    - food
```

## 🔍 查看处理结果

### 方法1：启动Web UI

```bash
# 启动API服务（使用uv）
uv run uvicorn app.main:app --reload

# 启动Web界面（使用uv）
uv run streamlit run ui/app.py

# 访问 http://localhost:8501
```

### 方法2：直接查询数据库

```python
import sqlite3

conn = sqlite3.connect('data/vibe_photos.db')
cursor = conn.cursor()

# 查看处理统计
cursor.execute("""
    SELECT process_status, COUNT(*) 
    FROM images 
    GROUP BY process_status
""")
print(cursor.fetchall())

# 查看检测到的物体
cursor.execute("""
    SELECT object_class, COUNT(*) as count
    FROM detections
    GROUP BY object_class
    ORDER BY count DESC
    LIMIT 10
""")
print("Top 10 检测到的物体:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]}")
```

## ❓ 常见问题

### Q: 如何清除所有处理结果重新开始？

```bash
# 方法1：完全重置（删除所有）
rm -rf data/*
rm -rf cache/*

# 方法2：保留缓存重置（推荐，下次处理更快）
rm -rf data/*
# cache保留，可以复用
```

### Q: 如何处理特定子目录？

修改 `config.yaml`:
```yaml
dataset:
  directory: "samples/Beijing, October 29, 2025"
```

### Q: 如何跳过去重检查？

```yaml
preprocessing:
  deduplication:
    enabled: false
```

### Q: 如何查看重复的图片？

```sql
-- 在数据库中查询
SELECT * FROM images WHERE process_status = 'duplicate';
```

## 📊 性能参考

基于测试环境（8核CPU，16GB内存）：

### 首次处理（无缓存）
- 预处理速度：~50张/分钟
- SigLIP+BLIP识别：~15张/分钟（CPU）
- PaddleOCR：~30张/分钟
- **整体处理：~10-15张/分钟**

### 缓存复用（第二次运行）
- 读取缓存：~200张/分钟
- 数据库写入：~150张/分钟
- **整体处理：~100+张/分钟**（10倍提升！）

### GPU加速
- GPU可提升检测速度2-3倍
- 但缓存复用的提升更明显（10倍）

### 跨版本复用
- Phase 1 → Phase 2：节省90%的预处理时间
- Phase 2 → Phase 3：所有缓存可直接使用

## 🎯 下一步

数据集处理完成后，您可以：

1. 使用搜索功能查找特定物体
2. 浏览检测结果和提取的文本
3. 导出搜索结果
4. 继续添加新图片（增量处理）

---

记住：PoC阶段专注于**验证核心功能**，不追求完美的用户体验。
