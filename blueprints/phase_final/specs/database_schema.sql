-- Vibe Photos Phase Final 数据库设计
-- 支持SQLite和PostgreSQL

-- ==========================================
-- 核心表
-- ==========================================

-- 照片主表
CREATE TABLE photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- 文件信息
    path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    file_hash TEXT,  -- SHA256 用于去重
    file_size INTEGER,
    
    -- 图像属性
    width INTEGER,
    height INTEGER,
    format TEXT,  -- jpg, png, etc
    
    -- 时间信息
    taken_at TIMESTAMP,  -- EXIF时间
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- AI识别结果
    ai_category TEXT,
    ai_confidence REAL,
    ai_subcategory TEXT,
    ai_brand TEXT,
    ai_model TEXT,  -- 具体型号
    ai_attributes JSON,  -- {color, size, style, etc}
    
    -- OCR结果
    ocr_text TEXT,
    ocr_language TEXT,
    
    -- 用户数据
    user_label TEXT,
    user_tags TEXT,  -- 逗号分隔的标签
    user_notes TEXT,
    is_favorite BOOLEAN DEFAULT FALSE,
    is_hidden BOOLEAN DEFAULT FALSE,
    
    -- 状态
    process_status TEXT DEFAULT 'pending',  -- pending, processing, completed, failed
    needs_review BOOLEAN DEFAULT FALSE,
    review_reason TEXT,
    
    -- 向量嵌入（用于相似搜索）
    embedding BLOB,  -- 序列化的浮点数组
    
    -- 索引提示
    CHECK (ai_confidence >= 0 AND ai_confidence <= 1),
    CHECK (process_status IN ('pending', 'processing', 'completed', 'failed'))
);

-- ==========================================
-- 检测结果表
-- ==========================================

-- 物体检测结果
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    
    -- 检测信息
    object_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    
    -- 边界框
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_width INTEGER,
    bbox_height INTEGER,
    
    -- 额外属性
    attributes JSON,
    
    -- 模型信息
    model_name TEXT,
    model_version TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CHECK (confidence >= 0 AND confidence <= 1)
);

-- ==========================================
-- 标注管理
-- ==========================================

-- 标注历史
CREATE TABLE annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    
    -- AI预测
    ai_prediction TEXT,
    ai_confidence REAL,
    ai_suggestions JSON,  -- [{label, score}, ...]
    
    -- 人工标注
    user_label TEXT NOT NULL,
    user_confirmed BOOLEAN DEFAULT TRUE,
    
    -- 批量应用
    batch_applied BOOLEAN DEFAULT FALSE,
    batch_group_id TEXT,  -- 批量操作组ID
    
    -- 学习相关
    used_for_training BOOLEAN DEFAULT FALSE,
    training_batch_id TEXT,
    
    -- 时间戳
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT,  -- 用户ID（如果有多用户）
    
    CHECK (ai_confidence >= 0 AND ai_confidence <= 1)
);

-- ==========================================
-- Few-Shot学习
-- ==========================================

-- 自定义产品库
CREATE TABLE custom_products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- 产品信息
    name TEXT UNIQUE NOT NULL,
    category TEXT NOT NULL,
    subcategory TEXT,
    brand TEXT,
    
    -- 学习信息
    sample_count INTEGER DEFAULT 0,
    min_confidence_threshold REAL DEFAULT 0.7,
    
    -- 模型数据
    prototype_vector BLOB,  -- 原型向量
    model_path TEXT,  -- 模型文件路径（如果有）
    
    -- 性能指标
    accuracy REAL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 元数据
    description TEXT,
    attributes JSON,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT,
    
    CHECK (min_confidence_threshold >= 0 AND min_confidence_threshold <= 1),
    CHECK (accuracy >= 0 AND accuracy <= 1)
);

-- 产品学习样本
CREATE TABLE product_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL REFERENCES custom_products(id) ON DELETE CASCADE,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    
    -- 样本信息
    is_positive BOOLEAN DEFAULT TRUE,  -- 正样本/负样本
    feature_vector BLOB,  -- 特征向量
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(product_id, photo_id)
);

-- ==========================================
-- 搜索和组织
-- ==========================================

-- 照片集合
CREATE TABLE collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    
    -- 集合类型
    type TEXT DEFAULT 'manual',  -- manual, smart, timeline
    
    -- 智能集合规则
    smart_rules JSON,  -- {category: "电子产品", date_range: {...}}
    
    -- 统计
    photo_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 照片-集合关联
CREATE TABLE collection_photos (
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    
    -- 排序和元数据
    sort_order INTEGER,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (collection_id, photo_id)
);

-- 搜索历史
CREATE TABLE search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    query TEXT NOT NULL,
    filters JSON,
    result_count INTEGER,
    
    -- 性能指标
    response_time_ms INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT
);

-- ==========================================
-- 系统管理
-- ==========================================

-- 处理队列
CREATE TABLE processing_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER REFERENCES photos(id) ON DELETE CASCADE,
    
    task_type TEXT NOT NULL,  -- detect, ocr, embed, thumbnail
    priority INTEGER DEFAULT 5,
    status TEXT DEFAULT 'pending',  -- pending, processing, completed, failed
    
    -- 重试机制
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- 错误信息
    error_message TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    CHECK (priority >= 1 AND priority <= 10),
    CHECK (status IN ('pending', 'processing', 'completed', 'failed'))
);

-- 系统配置
CREATE TABLE config (
    key TEXT PRIMARY KEY,
    value TEXT,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================
-- 索引优化
-- ==========================================

-- 照片表索引
CREATE INDEX idx_photos_path ON photos(path);
CREATE INDEX idx_photos_hash ON photos(file_hash);
CREATE INDEX idx_photos_category ON photos(ai_category);
CREATE INDEX idx_photos_user_label ON photos(user_label);
CREATE INDEX idx_photos_taken_at ON photos(taken_at);
CREATE INDEX idx_photos_imported_at ON photos(imported_at);
CREATE INDEX idx_photos_needs_review ON photos(needs_review);
CREATE INDEX idx_photos_status ON photos(process_status);

-- 检测结果索引
CREATE INDEX idx_detections_photo_id ON detections(photo_id);
CREATE INDEX idx_detections_class ON detections(object_class);

-- 标注索引
CREATE INDEX idx_annotations_photo_id ON annotations(photo_id);
CREATE INDEX idx_annotations_user_label ON annotations(user_label);
CREATE INDEX idx_annotations_training ON annotations(used_for_training);

-- 集合索引
CREATE INDEX idx_collection_photos_collection ON collection_photos(collection_id);
CREATE INDEX idx_collection_photos_photo ON collection_photos(photo_id);

-- 队列索引
CREATE INDEX idx_queue_status ON processing_queue(status, priority);
CREATE INDEX idx_queue_photo_id ON processing_queue(photo_id);

-- ==========================================
-- 视图
-- ==========================================

-- 需要审核的照片视图
CREATE VIEW photos_need_review AS
SELECT 
    p.id,
    p.path,
    p.filename,
    p.ai_category,
    p.ai_confidence,
    p.review_reason,
    p.imported_at
FROM photos p
WHERE p.needs_review = TRUE
  AND p.process_status = 'completed'
ORDER BY p.imported_at DESC;

-- 照片统计视图
CREATE VIEW photo_statistics AS
SELECT 
    COUNT(*) as total_photos,
    COUNT(CASE WHEN ai_category IS NOT NULL THEN 1 END) as categorized_photos,
    COUNT(CASE WHEN user_label IS NOT NULL THEN 1 END) as labeled_photos,
    COUNT(CASE WHEN needs_review = TRUE THEN 1 END) as review_needed,
    COUNT(CASE WHEN is_favorite = TRUE THEN 1 END) as favorites,
    AVG(ai_confidence) as avg_confidence
FROM photos;

-- 分类统计视图
CREATE VIEW category_statistics AS
SELECT 
    ai_category as category,
    COUNT(*) as photo_count,
    AVG(ai_confidence) as avg_confidence,
    COUNT(CASE WHEN user_label IS NOT NULL THEN 1 END) as verified_count
FROM photos
WHERE ai_category IS NOT NULL
GROUP BY ai_category
ORDER BY photo_count DESC;

-- ==========================================
-- 触发器
-- ==========================================

-- 更新modified_at时间戳
CREATE TRIGGER update_photo_modified_time 
AFTER UPDATE ON photos
BEGIN
    UPDATE photos SET modified_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- 更新集合照片数
CREATE TRIGGER update_collection_count_insert
AFTER INSERT ON collection_photos
BEGIN
    UPDATE collections 
    SET photo_count = photo_count + 1,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.collection_id;
END;

CREATE TRIGGER update_collection_count_delete
AFTER DELETE ON collection_photos
BEGIN
    UPDATE collections 
    SET photo_count = photo_count - 1,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = OLD.collection_id;
END;

-- ==========================================
-- 初始数据
-- ==========================================

-- 默认配置
INSERT INTO config (key, value, description) VALUES
('auto_classify', 'true', '自动分类新照片'),
('confidence_threshold', '0.7', 'AI分类的置信度阈值'),
('enable_ocr', 'true', '启用OCR文字提取'),
('enable_few_shot', 'true', '启用Few-shot学习'),
('batch_size', '16', '批处理大小'),
('thumbnail_size', '512', '缩略图大小'),
('model_name', 'clip-vit-base-patch32', '默认使用的模型');

-- 默认集合
INSERT INTO collections (name, description, type) VALUES
('最近导入', '最近7天导入的照片', 'smart'),
('待审核', '需要人工确认的照片', 'smart'),
('收藏夹', '标记为收藏的照片', 'smart');
