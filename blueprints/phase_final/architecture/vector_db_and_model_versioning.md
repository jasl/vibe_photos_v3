# 向量数据库与模型版本管理方案

## 🗂 向量数据库架构

### 技术选型：PostgreSQL + pgvector

#### 选择理由
- **统一存储**：元数据与向量在同一数据库，避免同步问题
- **事务一致性**：ACID保证，数据安全可靠
- **原生SQL支持**：结合传统查询与向量搜索
- **简化运维**：单一系统，降低复杂度
- **足够性能**：对于百万级向量完全够用（我们只有3万张照片）

### 系统架构
```
┌────────────────────────────────────────────┐
│              应用层                         │
│   ┌──────────────────────────────────┐    │
│   │      Vector Service API          │    │
│   └──────────────────────────────────┘    │
└────────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌─────────┐ ┌─────────────┐ ┌──────────────┐
│  Write  │ │   Search    │ │   Update     │
│  Path   │ │   Path      │ │   Path       │
└─────────┘ └─────────────┘ └──────────────┘
    │            │               │
    ▼            ▼               ▼
┌────────────────────────────────────────────┐
│     PostgreSQL + pgvector (主存储)          │
│  ┌──────────────────────────────────────┐  │
│  │  Metadata Tables                     │  │
│  │  - photos, detections, annotations   │  │
│  └──────────────────────────────────────┘  │
│  ┌──────────────────────────────────────┐  │
│  │  Vector Storage (pgvector)           │  │
│  │  - HNSW index for fast search        │  │
│  │  - Cosine/L2 distance metrics        │  │
│  │  - Hybrid SQL+Vector queries         │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
                    │
            [可选扩展：百万级+]
                    ▼
┌────────────────────────────────────────────┐
│         Faiss Cache Layer (Optional)        │
│   仅当向量超过100万时考虑引入               │
└────────────────────────────────────────────┘
```

### pgvector 性能特性
```yaml
index_types:
  ivfflat:  # 倒排文件索引
    - 适合: 10万-100万向量
    - 构建快，查询快
    - 需要定期重建
    
  hnsw:  # 分层导航小世界图
    - 适合: 任意规模（推荐）
    - 构建慢，查询极快
    - 增量友好，无需重建
    - 我们的选择 ✅

performance_benchmarks:
  30k_vectors:  # 我们的规模
    - 索引构建: < 1分钟
    - 单次查询: < 20ms
    - 批量查询: < 100ms (50个)
    - 内存占用: < 500MB
    
  1m_vectors:  # 未来扩展
    - 索引构建: < 30分钟
    - 单次查询: < 100ms
    - 批量查询: < 500ms
    - 内存占用: < 8GB
```

## 📊 数据层设计

### 数据库Schema（使用pgvector）
```sql
-- 启用pgvector扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- 照片向量表（主表）
CREATE TABLE photo_embeddings (
    id BIGSERIAL PRIMARY KEY,
    photo_id BIGINT REFERENCES photos(id) ON DELETE CASCADE,
    
    -- 向量信息
    embedding_model TEXT NOT NULL,  -- 'clip-vit-base', 'dinov2', etc
    embedding_version TEXT NOT NULL,  -- 'v1.0.0', 'v1.1.0'
    embedding_dimension INT NOT NULL,  -- 512, 768, 1024
    
    -- 向量存储（pgvector原生类型）
    embedding vector(768) NOT NULL,  -- 可调整维度
    
    -- 元数据
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,  -- 软删除标记
    
    -- 约束
    UNIQUE(photo_id, embedding_model, embedding_version),
    CHECK (embedding_dimension > 0 AND embedding_dimension <= 4096)
);

-- 创建HNSW索引（推荐用于我们的规模）
CREATE INDEX photo_embeddings_hnsw_idx ON photo_embeddings 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 或使用IVFFlat索引（更快构建，适合频繁更新）
-- CREATE INDEX photo_embeddings_ivf_idx ON photo_embeddings 
-- USING ivfflat (embedding vector_l2_ops)
-- WITH (lists = 100);

-- 向量索引版本管理
CREATE TABLE vector_indices (
    id SERIAL PRIMARY KEY,
    index_name TEXT UNIQUE NOT NULL,
    index_type TEXT NOT NULL,  -- 'IVF', 'HNSW', 'Flat'
    
    -- 索引配置
    config JSONB NOT NULL,  -- {"nlist": 4096, "nprobe": 64}
    total_vectors INT DEFAULT 0,
    dimension INT NOT NULL,
    
    -- 模型信息
    base_model TEXT NOT NULL,
    model_version TEXT NOT NULL,
    
    -- 文件路径
    index_path TEXT NOT NULL,
    backup_path TEXT,
    
    -- 状态管理
    status TEXT DEFAULT 'building',  -- building, active, deprecated
    is_primary BOOLEAN DEFAULT FALSE,
    
    -- 性能指标
    build_time_seconds FLOAT,
    search_latency_ms FLOAT,
    recall_at_10 FLOAT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP
);

-- 增量更新日志
CREATE TABLE vector_updates (
    id BIGSERIAL PRIMARY KEY,
    photo_id BIGINT REFERENCES photos(id),
    
    operation TEXT NOT NULL,  -- 'insert', 'update', 'delete'
    old_vector_id BIGINT,
    new_vector_id BIGINT,
    
    -- 批次信息
    batch_id UUID,
    batch_size INT,
    
    -- 状态
    status TEXT DEFAULT 'pending',  -- pending, processing, completed, failed
    error_message TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP
);
```

## 🔄 增量更新机制

### 1. 实时更新策略（pgvector原生）
```python
# services/vector_updater.py
from typing import List, Tuple, Optional
import numpy as np
from datetime import datetime
import asyncpg

class PgVectorUpdater:
    """PostgreSQL向量更新服务"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db = db_pool
        
    async def add_vector(self, photo_id: int, embedding: np.ndarray, 
                        model_name: str = "clip-vit-base") -> int:
        """添加新向量"""
        async with self.db.acquire() as conn:
            # pgvector自动处理索引更新
            vector_id = await conn.fetchval("""
                INSERT INTO photo_embeddings 
                (photo_id, embedding, embedding_model, embedding_version, embedding_dimension)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """, photo_id, embedding.tolist(), model_name, "v1.0.0", len(embedding))
            
            # 记录操作日志
            await self.log_update(conn, 'insert', photo_id, vector_id)
            
        return vector_id
    
    async def update_vector(self, photo_id: int, new_embedding: np.ndarray) -> int:
        """更新现有向量（原子操作）"""
        async with self.db.acquire() as conn:
            async with conn.transaction():
                # 软删除旧向量
                await conn.execute("""
                    UPDATE photo_embeddings 
                    SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
                    WHERE photo_id = $1 AND is_active = TRUE
                """, photo_id)
                
                # 插入新向量
                new_id = await conn.fetchval("""
                    INSERT INTO photo_embeddings 
                    (photo_id, embedding, embedding_model, embedding_version, embedding_dimension)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                """, photo_id, new_embedding.tolist(), "clip-vit-base", "v1.0.0", len(new_embedding))
                
                # 记录更新
                await self.log_update(conn, 'update', photo_id, new_id)
                
        return new_id
    
    async def batch_add_vectors(self, vectors: List[Tuple[int, np.ndarray]]) -> List[int]:
        """批量添加向量（优化性能）"""
        async with self.db.acquire() as conn:
            # 使用COPY命令批量插入，性能最优
            result = await conn.copy_records_to_table(
                'photo_embeddings',
                records=[(pid, emb.tolist(), "clip-vit-base", "v1.0.0", len(emb)) 
                        for pid, emb in vectors],
                columns=['photo_id', 'embedding', 'embedding_model', 
                        'embedding_version', 'embedding_dimension']
            )
            
            return result
    
    async def search_similar(self, query_embedding: np.ndarray, 
                            limit: int = 10) -> List[dict]:
        """相似向量搜索"""
        async with self.db.acquire() as conn:
            # pgvector原生相似度搜索
            results = await conn.fetch("""
                SELECT 
                    p.photo_id,
                    p.embedding <=> $1 as distance,
                    ph.path,
                    ph.category
                FROM photo_embeddings p
                JOIN photos ph ON p.photo_id = ph.id
                WHERE p.is_active = TRUE
                ORDER BY p.embedding <=> $1
                LIMIT $2
            """, query_embedding.tolist(), limit)
            
            return [dict(r) for r in results]
    
    async def optimize_index(self):
        """优化向量索引"""
        async with self.db.acquire() as conn:
            # REINDEX优化索引性能
            await conn.execute("REINDEX INDEX CONCURRENTLY photo_embeddings_hnsw_idx")
            
            # 清理软删除的向量
            await conn.execute("""
                DELETE FROM photo_embeddings 
                WHERE is_active = FALSE 
                AND updated_at < CURRENT_TIMESTAMP - INTERVAL '7 days'
            """)
            
            # 更新统计信息
            await conn.execute("ANALYZE photo_embeddings")
```

### 2. 批量更新优化
```python
# services/batch_updater.py
class BatchVectorUpdater:
    """批量向量更新优化"""
    
    async def batch_update(self, updates: List[Tuple[int, np.ndarray]]):
        """批量更新向量"""
        batch_id = uuid.uuid4()
        
        # 1. 批量验证
        valid_updates = await self.validate_batch(updates)
        
        # 2. 事务处理
        async with self.db.transaction():
            # 批量写入PostgreSQL
            vector_ids = await self.bulk_insert_postgres(valid_updates, batch_id)
            
            # 批量更新Faiss
            vectors = np.vstack([u[1] for u in valid_updates])
            self.add_batch_to_faiss(vectors)
            
            # 记录批次日志
            await self.log_batch_update(batch_id, len(valid_updates))
        
        # 3. 异步后处理
        asyncio.create_task(self.post_process_batch(batch_id))
        
        return {'batch_id': batch_id, 'processed': len(valid_updates)}
    
    async def post_process_batch(self, batch_id: str):
        """批次后处理"""
        # 更新索引统计
        await self.update_index_statistics()
        
        # 触发索引优化
        if await self.should_optimize():
            await self.optimize_index()
        
        # 清理旧向量
        await self.cleanup_old_vectors(batch_id)
```

### 3. 版本迁移策略
```python
# services/version_migration.py
class VectorVersionMigrator:
    """向量版本迁移服务"""
    
    async def migrate_to_new_model(self, 
                                   old_model: str, 
                                   new_model: str,
                                   batch_size: int = 1000):
        """迁移到新模型版本"""
        
        # 1. 创建新索引
        new_index = await self.create_new_index(new_model)
        
        # 2. 分批迁移
        total_photos = await self.get_photo_count()
        
        for offset in range(0, total_photos, batch_size):
            # 获取批次照片
            photos = await self.get_photos_batch(offset, batch_size)
            
            # 生成新向量
            new_vectors = await self.generate_embeddings(photos, new_model)
            
            # 写入新索引
            await self.add_to_new_index(new_index, new_vectors)
            
            # 更新进度
            progress = (offset + batch_size) / total_photos
            await self.update_migration_progress(new_model, progress)
        
        # 3. 切换索引
        await self.atomic_switch_index(old_model, new_model)
        
        # 4. 验证迁移
        await self.validate_migration(old_model, new_model)
```

## 🏷 模型版本管理

### 1. 模型注册表
```sql
-- 模型版本注册表
CREATE TABLE model_registry (
    id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,  -- 'clip-detector', 'rtmdet', 'few-shot-v1'
    model_type TEXT NOT NULL,  -- 'detection', 'embedding', 'classification'
    
    -- 版本信息
    version TEXT NOT NULL,  -- 语义版本号 '1.2.3'
    version_tag TEXT,  -- 'stable', 'beta', 'deprecated'
    
    -- 模型文件
    model_path TEXT NOT NULL,
    config_path TEXT,
    checkpoint_path TEXT,
    file_size_mb FLOAT,
    
    -- 性能指标
    metrics JSONB,  -- {"accuracy": 0.92, "f1": 0.89, "latency_ms": 45}
    benchmark_results JSONB,
    
    -- 依赖关系
    parent_version TEXT,  -- 基于哪个版本
    dependencies JSONB,  -- {"transformers": "4.30.0"}
    
    -- 部署信息
    deployment_status TEXT DEFAULT 'testing',  -- testing, staging, production
    deployed_at TIMESTAMP,
    deprecated_at TIMESTAMP,
    
    -- 元数据
    description TEXT,
    changelog TEXT,
    created_by TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(model_name, version)
);

-- 模型部署历史
CREATE TABLE model_deployments (
    id SERIAL PRIMARY KEY,
    model_id INT REFERENCES model_registry(id),
    
    environment TEXT NOT NULL,  -- 'dev', 'staging', 'prod'
    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deployed_by TEXT,
    
    -- 部署配置
    config JSONB,
    resources JSONB,  -- {"cpu": 4, "memory": "8GB", "gpu": "V100"}
    
    -- 回滚信息
    rollback_from INT REFERENCES model_deployments(id),
    rollback_reason TEXT,
    rollback_at TIMESTAMP
);

-- A/B测试配置
CREATE TABLE model_ab_tests (
    id SERIAL PRIMARY KEY,
    test_name TEXT UNIQUE NOT NULL,
    
    -- 测试模型
    model_a_id INT REFERENCES model_registry(id),
    model_b_id INT REFERENCES model_registry(id),
    
    -- 流量分配
    traffic_split FLOAT DEFAULT 0.5,  -- A模型流量比例
    
    -- 测试配置
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    
    -- 测试结果
    metrics_a JSONB,
    metrics_b JSONB,
    winner TEXT,  -- 'model_a', 'model_b', 'no_difference'
    
    status TEXT DEFAULT 'running'  -- 'planning', 'running', 'completed'
);
```

### 2. 模型版本控制
```python
# services/model_versioning.py
from semantic_version import Version
import hashlib

class ModelVersionManager:
    """模型版本管理器"""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.storage = ModelStorage()
    
    async def register_model(self, 
                            model_name: str,
                            model_path: str,
                            metrics: dict,
                            auto_version: bool = True):
        """注册新模型版本"""
        
        # 1. 计算模型哈希
        model_hash = self.calculate_model_hash(model_path)
        
        # 2. 检查是否已存在
        if await self.model_exists(model_hash):
            raise ModelAlreadyExistsError(f"Model {model_hash} already registered")
        
        # 3. 自动版本号
        if auto_version:
            version = await self.get_next_version(model_name, metrics)
        
        # 4. 保存模型文件
        stored_path = await self.storage.save_model(
            model_path, 
            f"{model_name}/{version}"
        )
        
        # 5. 注册到数据库
        model_id = await self.registry.register(
            model_name=model_name,
            version=version,
            path=stored_path,
            metrics=metrics
        )
        
        # 6. 运行验证测试
        await self.validate_model(model_id)
        
        return model_id
    
    async def get_next_version(self, model_name: str, metrics: dict) -> str:
        """自动生成版本号"""
        current = await self.get_latest_version(model_name)
        
        if not current:
            return "1.0.0"
        
        version = Version(current)
        
        # 根据性能决定版本号
        current_metrics = await self.get_model_metrics(model_name, current)
        
        if metrics.get('accuracy', 0) > current_metrics.get('accuracy', 0) * 1.1:
            # 性能提升>10%，主版本号+1
            return str(version.next_major())
        elif metrics.get('accuracy', 0) > current_metrics.get('accuracy', 0):
            # 性能提升，次版本号+1
            return str(version.next_minor())
        else:
            # 补丁版本
            return str(version.next_patch())
    
    async def deploy_model(self, 
                          model_id: int, 
                          environment: str,
                          strategy: str = 'blue_green'):
        """部署模型"""
        
        if strategy == 'blue_green':
            return await self.blue_green_deploy(model_id, environment)
        elif strategy == 'canary':
            return await self.canary_deploy(model_id, environment)
        elif strategy == 'ab_test':
            return await self.ab_test_deploy(model_id, environment)
        else:
            return await self.direct_deploy(model_id, environment)
```

### 3. 模型热更新
```python
# services/model_hot_reload.py
class ModelHotReloader:
    """模型热更新服务"""
    
    def __init__(self):
        self.active_models = {}
        self.model_locks = {}
    
    async def hot_reload(self, model_name: str, new_version: str):
        """热更新模型"""
        
        # 1. 预加载新模型
        new_model = await self.preload_model(model_name, new_version)
        
        # 2. 获取写锁
        async with self.get_model_lock(model_name, mode='write'):
            # 3. 备份当前模型
            old_model = self.active_models.get(model_name)
            
            # 4. 原子切换
            self.active_models[model_name] = new_model
            
            # 5. 验证新模型
            try:
                await self.validate_loaded_model(model_name)
            except ValidationError as e:
                # 回滚
                self.active_models[model_name] = old_model
                raise ModelUpdateError(f"Validation failed: {e}")
            
            # 6. 清理旧模型
            if old_model:
                await self.cleanup_model(old_model)
        
        # 7. 更新路由
        await self.update_model_routing(model_name, new_version)
        
        return {"status": "success", "model": model_name, "version": new_version}
    
    async def validate_loaded_model(self, model_name: str):
        """验证加载的模型"""
        model = self.active_models[model_name]
        
        # 运行测试用例
        test_cases = await self.get_test_cases(model_name)
        for test in test_cases:
            result = model.predict(test.input)
            assert result.shape == test.expected_shape
            assert result.dtype == test.expected_dtype
```

## 🔄 备份与恢复机制

### 1. 简化的备份策略（单一数据源优势）
```python
# services/backup_service.py
class PgVectorBackupService:
    """PostgreSQL向量备份服务"""
    
    async def backup_vectors(self, backup_name: str):
        """备份向量数据（利用PostgreSQL原生备份）"""
        
        # 1. 使用pg_dump备份向量表
        backup_cmd = f"""
        pg_dump -h localhost -U user -d vibe_photos \
                -t photo_embeddings -t vector_updates \
                -f backups/{backup_name}_vectors.sql
        """
        await self.execute_backup(backup_cmd)
        
        # 2. 创建备份元数据
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'total_vectors': await self.get_vector_count(),
            'backup_file': f"{backup_name}_vectors.sql",
            'index_type': 'hnsw',
            'model_versions': await self.get_active_model_versions()
        }
        
        # 3. 可选：上传到S3
        if self.use_s3_backup:
            await self.upload_to_s3(backup_name, metadata)
        
        return metadata
    
    async def restore_from_backup(self, backup_name: str):
        """恢复向量数据"""
        
        # 1. 恢复PostgreSQL数据
        restore_cmd = f"""
        psql -h localhost -U user -d vibe_photos \
             -f backups/{backup_name}_vectors.sql
        """
        await self.execute_restore(restore_cmd)
        
        # 2. 重建索引（如果需要）
        await self.rebuild_indices()
        
        # 3. 验证数据完整性
        await self.validate_restoration()
    
    async def incremental_backup(self, since: datetime):
        """增量备份（仅备份变更）"""
        async with self.db.acquire() as conn:
            changes = await conn.fetch("""
                SELECT * FROM photo_embeddings 
                WHERE created_at > $1 OR updated_at > $1
            """, since)
            
            # 导出为JSON或其他格式
            await self.export_changes(changes)
```

## 🚀 可选性能优化：何时引入Faiss

### 触发条件
```yaml
consider_faiss_when:
  vector_count: > 1,000,000  # 超过百万向量
  query_latency: > 200ms     # 查询延迟过高
  concurrent_users: > 1000    # 高并发场景
  special_requirements:
    - 需要GPU加速
    - 需要特殊索引类型（PQ、LSH等）
    - 需要极致的批量查询性能

current_status:
  our_scale: 30,000 vectors  # 远低于阈值
  expected_latency: < 20ms   # pgvector足够快
  conclusion: "不需要Faiss"  # ✅
```

### 未来扩展路径（如需要）
```python
# services/hybrid_search.py (仅当规模超过百万时)
class HybridVectorSearch:
    """混合搜索策略：pgvector + Faiss缓存"""
    
    def __init__(self):
        self.pg_searcher = PgVectorSearcher()
        self.faiss_cache = None  # 延迟初始化
        
    async def search(self, query_vector, limit=10):
        # 默认使用pgvector
        if self.should_use_faiss():
            # 仅对热门查询使用Faiss缓存
            return await self.faiss_cached_search(query_vector, limit)
        else:
            # 常规查询直接用pgvector
            return await self.pg_searcher.search(query_vector, limit)
    
    def should_use_faiss(self):
        # 动态判断是否需要Faiss
        return (
            self.total_vectors > 1_000_000 or
            self.avg_query_latency > 200  # ms
        )
```

## 📊 监控与告警

### 监控指标
```python
# monitoring/vector_metrics.py
from prometheus_client import Histogram, Counter, Gauge

# 性能指标
vector_search_latency = Histogram(
    'vector_search_latency_seconds',
    'Vector search latency',
    ['index_type', 'query_type']
)

vector_update_latency = Histogram(
    'vector_update_latency_seconds',
    'Vector update latency',
    ['operation']
)

# 数据指标
total_vectors = Gauge(
    'total_vectors_count',
    'Total number of vectors',
    ['model', 'version']
)

index_size_bytes = Gauge(
    'index_size_bytes',
    'Index file size in bytes',
    ['index_name']
)

# 一致性指标
sync_lag_seconds = Gauge(
    'vector_sync_lag_seconds',
    'Sync lag between PostgreSQL and Faiss'
)

inconsistency_count = Counter(
    'vector_inconsistency_total',
    'Total inconsistencies detected',
    ['type']
)
```

## 📋 实施检查清单

### 向量数据库（pgvector为主）
- [ ] 安装PostgreSQL 14+
- [ ] 安装pgvector扩展
- [ ] 创建向量表和HNSW索引
- [ ] 实现向量CRUD操作
- [ ] 配置批量插入优化
- [ ] 实现混合查询（SQL + 向量）
- [ ] 设置自动清理任务
- [ ] 配置PostgreSQL备份策略
- [ ] 添加查询性能监控
- [ ] 准备Faiss扩展方案（预留）

### 模型版本管理
- [ ] 创建模型注册表
- [ ] 实现版本控制逻辑
- [ ] 配置热更新机制
- [ ] 设置A/B测试框架
- [ ] 实现蓝绿部署
- [ ] 配置回滚策略
- [ ] 添加模型验证流程
- [ ] 设置性能基准测试
