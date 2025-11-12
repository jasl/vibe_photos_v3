# 消息队列与任务编排方案

## 📬 技术选型

### 主选方案：Celery + Redis

#### 选择理由
- **成熟稳定**：Celery是Python生态最成熟的任务队列方案
- **易于集成**：与FastAPI无缝集成，原生Python支持
- **功能完整**：支持定时任务、任务链、重试、优先级等
- **监控友好**：Flower提供可视化监控界面
- **轻量部署**：Redis作为Broker，资源占用小

#### 架构设计
```
┌────────────────────────────────────────────┐
│                  FastAPI App                │
│  ┌──────────────────────────────────────┐  │
│  │     API Endpoints (异步接收)         │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
                      │
                      ▼ 提交任务
┌────────────────────────────────────────────┐
│              Redis (Message Broker)         │
│  ┌──────────────────────────────────────┐  │
│  │   Queue: high_priority (实时处理)    │  │
│  │   Queue: default (常规任务)          │  │
│  │   Queue: batch (批量处理)            │  │
│  │   Queue: learning (模型训练)         │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
                      │
                      ▼ 消费任务
┌────────────────────────────────────────────┐
│           Celery Workers Pool               │
│  ┌──────────────────────────────────────┐  │
│  │  Worker1: 图像检测 (GPU优化)         │  │
│  │  Worker2: OCR处理 (CPU密集)          │  │
│  │  Worker3: 向量计算 (并行处理)        │  │
│  │  Worker4: 数据同步 (I/O密集)         │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
```

## 🎯 任务类型定义

### 1. 实时任务（高优先级）
```python
# tasks/realtime.py
from celery import Task
from celery.exceptions import Retry

class ImageDetectionTask(Task):
    """单张图片实时检测"""
    name = 'detect.single'
    max_retries = 3
    default_retry_delay = 5
    
    def run(self, image_path: str, user_id: str):
        try:
            # SigLIP+BLIP检测
            detections = siglip_blip_detector.detect(image_path)
            
            # SigLIP分类
            category = siglip_classifier.classify(image_path)
            
            # 保存结果
            save_to_db(image_path, detections, category)
            
            # 实时推送结果
            websocket_notify(user_id, {'status': 'completed', 'path': image_path})
            
        except Exception as e:
            # 指数退避重试
            raise self.retry(exc=e, countdown=2 ** self.request.retries)

@celery.task(bind=True, queue='high_priority', priority=9)
def detect_single_image(self, image_path: str, user_id: str):
    return ImageDetectionTask().run(image_path, user_id)
```

### 2. 批量处理任务
```python
# tasks/batch.py
from celery import group, chain, chord

@celery.task(queue='batch')
def process_batch_import(folder_path: str, user_id: str):
    """批量导入照片"""
    
    # 1. 扫描文件
    images = scan_folder(folder_path)
    
    # 2. 创建任务组（并行处理）
    detection_group = group(
        detect_image.s(img) for img in images[:100]  # 限制并发
    )
    
    # 3. 串行任务链
    workflow = chain(
        validate_images.s(images),
        detection_group,
        aggregate_results.s(),
        update_vectors.s(),
        notify_completion.s(user_id)
    )
    
    return workflow.apply_async()

@celery.task(queue='batch')
def aggregate_results(results):
    """聚合批处理结果"""
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    # 更新统计
    update_stats(len(successful), len(failed))
    
    # 触发增量学习
    if len(successful) > 50:
        trigger_incremental_learning.delay(successful)
    
    return {
        'total': len(results),
        'success': len(successful),
        'failed': len(failed)
    }
```

### 3. 模型训练任务
```python
# tasks/learning.py
@celery.task(queue='learning', time_limit=3600)  # 1小时超时
def train_few_shot_model(samples: List[dict], model_name: str):
    """Few-shot学习任务"""
    
    # 获取独占锁，避免并发训练
    with redis_lock(f'training:{model_name}', timeout=3600):
        # 准备训练数据
        X, y = prepare_training_data(samples)
        
        # 训练模型
        model = FewShotLearner()
        model.fit(X, y)
        
        # 验证性能
        metrics = evaluate_model(model)
        
        if metrics['accuracy'] > 0.8:
            # 保存模型
            model_path = save_model(model, model_name)
            
            # 版本管理
            register_model_version(model_name, model_path, metrics)
            
            # 热更新模型
            hot_reload_model.delay(model_name, model_path)
        
        return metrics
```

## 🔄 任务调度策略

### 1. 优先级管理
```python
# config/celery_config.py
from kombu import Queue, Exchange

task_routes = {
    'detect.single': {'queue': 'high_priority', 'priority': 9},
    'batch.*': {'queue': 'batch', 'priority': 5},
    'learning.*': {'queue': 'learning', 'priority': 3},
    'sync.*': {'queue': 'default', 'priority': 1},
}

# 队列定义
task_queues = [
    Queue('high_priority', Exchange('high_priority'), 
          routing_key='high', priority=10),
    Queue('batch', Exchange('batch'), 
          routing_key='batch', priority=5),
    Queue('learning', Exchange('learning'), 
          routing_key='learning', priority=3),
    Queue('default', Exchange('default'), 
          routing_key='default', priority=1),
]

# 任务确认机制
task_acks_late = True  # 任务完成后才确认
task_reject_on_worker_lost = True  # Worker丢失时拒绝任务
```

### 2. 重试策略
```python
# utils/retry_policy.py
from celery import Task
from celery.exceptions import MaxRetriesExceededError

class RetryableTask(Task):
    """可重试任务基类"""
    
    autoretry_for = (ConnectionError, TimeoutError)
    retry_kwargs = {'max_retries': 3}
    retry_backoff = True  # 指数退避
    retry_backoff_max = 600  # 最大退避时间10分钟
    retry_jitter = True  # 添加随机抖动

@celery.task(base=RetryableTask)
def process_with_retry(image_path):
    """带重试的处理任务"""
    try:
        result = heavy_processing(image_path)
        return result
    except MaxRetriesExceededError:
        # 重试失败，进入死信队列
        send_to_dlq(image_path)
        notify_admin(f"Processing failed: {image_path}")
```

### 3. 任务编排模式
```python
# workflows/complex_workflow.py
from celery import chord, group, chain

def import_and_learn_workflow(folder_path: str):
    """复杂的导入和学习工作流"""
    
    # 阶段1：并行检测
    detection_tasks = group([
        detect_objects.s(img),
        extract_text.s(img),
        calculate_embedding.s(img)
    ] for img in get_images(folder_path))
    
    # 阶段2：聚合结果
    aggregate = aggregate_detection_results.s()
    
    # 阶段3：条件分支
    def route_by_confidence(results):
        high_conf = [r for r in results if r['confidence'] > 0.8]
        low_conf = [r for r in results if r['confidence'] < 0.5]
        
        if high_conf:
            auto_label.delay(high_conf)
        if low_conf:
            queue_for_review.delay(low_conf)
    
    # 组合工作流
    workflow = chord(detection_tasks)(aggregate | route_by_confidence)
    
    return workflow
```

## 📊 监控与可观测性

### 1. Flower监控配置
```python
# monitoring/flower_config.py
from flower import Flower

flower_config = {
    'broker': 'redis://localhost:6379/0',
    'port': 5555,
    'basic_auth': ['admin:secure_password'],
    'persistent': True,
    'db': 'flower.db',
    'max_tasks': 10000,
    'enable_events': True
}

# 自定义监控指标
custom_metrics = {
    'task_duration': histogram('task_duration_seconds'),
    'task_success_rate': gauge('task_success_rate'),
    'queue_length': gauge('queue_length'),
    'worker_utilization': gauge('worker_utilization')
}
```

### 2. 性能监控
```python
# monitoring/metrics.py
import time
from prometheus_client import Counter, Histogram, Gauge

# Prometheus指标
task_counter = Counter('celery_tasks_total', 'Total tasks', ['task', 'status'])
task_duration = Histogram('celery_task_duration_seconds', 'Task duration', ['task'])
queue_size = Gauge('celery_queue_size', 'Queue size', ['queue'])

@celery.task(bind=True)
def monitored_task(self, *args, **kwargs):
    """带监控的任务装饰器"""
    start_time = time.time()
    
    try:
        result = actual_task(*args, **kwargs)
        task_counter.labels(task=self.name, status='success').inc()
        return result
    except Exception as e:
        task_counter.labels(task=self.name, status='failure').inc()
        raise
    finally:
        duration = time.time() - start_time
        task_duration.labels(task=self.name).observe(duration)
```

## 🚦 流量控制

### 1. 速率限制
```python
# config/rate_limits.py
rate_limits = {
    'detect.single': '100/m',  # 每分钟100次
    'batch.*': '10/m',  # 每分钟10个批次
    'learning.*': '1/h',  # 每小时1次训练
}

# 动态速率调整
@celery.task
def adjust_rate_limits():
    """根据系统负载动态调整速率"""
    cpu_usage = get_cpu_usage()
    memory_usage = get_memory_usage()
    
    if cpu_usage > 80 or memory_usage > 80:
        # 降低速率
        celery.control.rate_limit('detect.single', '50/m')
    elif cpu_usage < 30 and memory_usage < 30:
        # 提高速率
        celery.control.rate_limit('detect.single', '200/m')
```

### 2. 背压处理
```python
# utils/backpressure.py
from celery import signals

@signals.task_prerun.connect
def check_backpressure(sender=None, task_id=None, task=None, **kwargs):
    """任务执行前检查背压"""
    queue_size = get_queue_size(task.queue)
    
    if queue_size > 1000:
        # 队列过长，拒绝新任务
        raise QueueOverloadError(f"Queue {task.queue} is overloaded")
    
    if queue_size > 500:
        # 警告级别，记录日志
        logger.warning(f"Queue {task.queue} size: {queue_size}")
```

## 🔧 部署配置

### 1. Docker Compose配置
```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  celery_worker:
    build: .
    command: celery -A app.celery worker -Q high_priority,default,batch -c 4
    depends_on:
      - redis
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0
    volumes:
      - ./photos:/photos
      - ./models:/models
    deploy:
      replicas: 2  # 2个worker实例

  celery_beat:
    build: .
    command: celery -A app.celery beat -l INFO
    depends_on:
      - redis
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0

  flower:
    build: .
    command: celery -A app.celery flower --port=5555
    ports:
      - "5555:5555"
    depends_on:
      - redis
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0

volumes:
  redis_data:
```

### 2. 生产环境配置
```python
# config/production.py
CELERY_CONFIG = {
    'broker_url': 'redis://redis-cluster:6379/0',
    'result_backend': 'redis://redis-cluster:6379/1',
    
    # 持久化
    'task_serializer': 'json',
    'result_serializer': 'json',
    'accept_content': ['json'],
    
    # 性能优化
    'worker_prefetch_multiplier': 4,
    'worker_max_tasks_per_child': 1000,
    'broker_pool_limit': 10,
    
    # 容错
    'task_acks_late': True,
    'task_reject_on_worker_lost': True,
    'task_publish_retry': True,
    'task_publish_retry_policy': {
        'max_retries': 3,
        'interval_start': 0,
        'interval_step': 0.2,
        'interval_max': 0.5,
    }
}
```

## 📋 实施清单

- [ ] 安装Redis服务器
- [ ] 配置Celery基础架构
- [ ] 实现任务类型（实时、批量、学习）
- [ ] 设置队列优先级
- [ ] 配置重试策略
- [ ] 部署Flower监控
- [ ] 实现速率限制
- [ ] 添加Prometheus指标
- [ ] 配置日志聚合
- [ ] 测试容错机制
- [ ] 编写任务编排示例
- [ ] 性能基准测试
