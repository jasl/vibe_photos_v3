# 🔧 AI代码规范 - Vibe Photos编码标准

> 本文档为Coding AI提供详细的代码规范和最佳实践，确保代码质量和一致性

## 🌐 语言使用规范

### 核心原则：代码英文，文档中文

| 文件类型 | 语言要求 | 说明 | 示例 |
|---------|---------|------|------|
| **源代码文件** | 纯英文 | 所有代码、注释、文档字符串必须使用英文 | `.py`, `.js`, `.yaml` |
| **文档文件** | 中文 | 面向用户的文档使用中文 | `.md` 文档 |
| **配置文件** | 英文 | 配置键值对使用英文 | `config.yaml`, `settings.json` |
| **测试文件** | 英文 | 测试代码和注释使用英文 | `test_*.py` |
| **提交信息** | 中文/英文 | 可以使用中文说明，但类型标识用英文 | `feat:`, `fix:`, `docs:` |

### 源代码英文规范

```python
# ✅ 正确示例 - 全英文
class ImageDetector:
    """
    Image detection module using SigLIP and BLIP models.
    
    This module provides functionality to detect and classify
    images using state-of-the-art AI models.
    """
    
    def detect_objects(self, image_path: Path) -> Dict:
        """
        Detect objects in the given image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing detection results
        """
        # Check if file exists
        if not image_path.exists():
            logger.error(f"Image file not found: {image_path}")
            return {"error": "File not found"}
        
        # Process the image
        result = self._process_image(image_path)
        
        return result

# ❌ 错误示例 - 混用中文
class ImageDetector:
    """
    图像检测模块  # 错误：使用了中文
    """
    
    def detect_objects(self, image_path: Path) -> Dict:
        # 检查文件是否存在  # 错误：注释使用了中文
        if not image_path.exists():
            logger.error(f"图片未找到: {image_path}")  # 错误：日志信息使用了中文
            return {"error": "文件不存在"}  # 错误：错误信息使用了中文
```

### 文档文件中文规范

```markdown
# ✅ 正确示例 - 文档使用中文

## 图像检测模块使用说明

本模块提供了强大的图像检测功能，支持以下特性：
- 多语言分类（支持中文标签）
- 批量处理
- 自动缓存

### 使用示例
\```python
# 代码部分仍然保持英文
detector = ImageDetector()
result = detector.detect("image.jpg")
\```
```

### 特殊情况处理

1. **用户界面文本**：存储在独立的本地化文件中
   ```python
   # messages_zh.py
   MESSAGES = {
       "welcome": "欢迎使用Vibe Photos",
       "processing": "正在处理图片...",
       "complete": "处理完成"
   }
   
   # main.py (英文)
   from locales.messages_zh import MESSAGES
   print(MESSAGES["welcome"])  # Output Chinese text
   ```

2. **配置文件注释**：使用英文
   ```yaml
   # config.yaml
   # Database configuration
   database:
     host: localhost  # Database host address
     port: 5432      # PostgreSQL default port
   ```

3. **日志输出**：关键信息用英文，用户提示可本地化
   ```python
   # System logs in English
   logger.info("Starting image processing")
   logger.error("Database connection failed")
   
   # User messages can be localized
   print(MESSAGES["processing"])  # 显示中文给用户
   ```

### 命名规范对照表

| 概念 | 英文命名 | 说明 |
|------|---------|------|
| 检测器 | detector | 不用 jiance_qi |
| 处理器 | processor | 不用 chuli_qi |
| 数据库 | database | 不用 shuju_ku |
| 搜索 | search | 不用 sousuo |
| 图像 | image | 不用 tupian |
| 分类 | category/classify | 不用 fenlei |
| 标签 | label/tag | 不用 biaoqian |
| 用户 | user | 不用 yonghu |

## 📐 Python代码规范

### 文件组织结构
```python
"""
模块文档字符串 - 简要说明模块功能
"""
# 标准库导入
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

# 第三方库导入
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 本地模块导入
from .core import detector
from .utils import logger

# 模块级常量（全大写）
DEFAULT_BATCH_SIZE = 16
MAX_IMAGE_SIZE = (1920, 1080)

# 模块级变量（小写）
_cache = {}
logger = logger.get_logger(__name__)

# 类定义
class ImageProcessor:
    """类文档字符串"""
    pass

# 函数定义
def process_image(image_path: Path) -> Dict:
    """函数文档字符串"""
    pass

# 主程序入口
if __name__ == "__main__":
    main()
```

### 命名规范
```python
# ✅ 正确的命名示例

# 变量名：小写，下划线分隔，描述性
user_input = "search query"
is_valid = True
has_permission = False
image_count = 42

# 函数名：小写，下划线分隔，动词开头
def process_image(image_path: Path) -> Dict:
    pass

def validate_input(data: str) -> bool:
    pass

def get_user_settings(user_id: int) -> Dict:
    pass

# 类名：大驼峰，名词
class ImageDetector:
    pass

class DatabaseManager:
    pass

# 常量：全大写，下划线分隔
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT = 30
API_VERSION = "1.0.0"

# 私有成员：单下划线前缀
class MyClass:
    def __init__(self):
        self._internal_state = {}
    
    def _private_method(self):
        pass

# ❌ 避免的命名
# 单字母变量（除了循环计数器）
x = process()  # 错误
result = process()  # 正确

# 缩写不清晰
def calc_img_sim():  # 错误
def calculate_image_similarity():  # 正确

# 匈牙利命名法
strName = "john"  # 错误
name = "john"  # 正确
```

### 类型注解规范
```python
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from pathlib import Path
import numpy as np

# 基础类型注解
def process_text(text: str) -> str:
    return text.upper()

# 可选类型
def search(
    query: str,
    limit: Optional[int] = None,
    category: Optional[str] = None
) -> List[Dict]:
    pass

# 联合类型
def load_image(source: Union[str, Path, bytes]) -> np.ndarray:
    pass

# 复杂类型
def batch_process(
    images: List[Path],
    processor: Callable[[Path], Dict],
    options: Dict[str, Any] = None
) -> List[Dict[str, Union[str, float]]]:
    pass

# 返回多个值
def detect_objects(image: np.ndarray) -> Tuple[List[str], List[float]]:
    labels = ["cat", "dog"]
    scores = [0.9, 0.8]
    return labels, scores

# 使用TypedDict定义复杂结构
from typing import TypedDict

class DetectionResult(TypedDict):
    category: str
    confidence: float
    bbox: List[int]
    metadata: Optional[Dict]

def detect(image_path: Path) -> DetectionResult:
    return {
        "category": "electronic",
        "confidence": 0.95,
        "bbox": [10, 20, 100, 200],
        "metadata": None
    }
```

### 函数设计原则
```python
# 1. 单一职责原则 - 每个函数只做一件事
# ✅ 好的示例
def load_image(path: Path) -> np.ndarray:
    """加载图像文件"""
    return cv2.imread(str(path))

def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """调整图像大小"""
    return cv2.resize(image, size)

# ❌ 不好的示例
def load_and_process_image(path: Path) -> np.ndarray:
    """加载并处理图像（做了太多事）"""
    image = cv2.imread(str(path))
    image = cv2.resize(image, (224, 224))
    image = normalize(image)
    return image

# 2. 早期返回模式 - 尽早处理错误情况
def process_data(data: Optional[Dict]) -> Dict:
    # 早期验证
    if data is None:
        logger.warning("No data provided")
        return {}
    
    if not data.get("images"):
        logger.warning("No images in data")
        return {"error": "No images"}
    
    # 主要逻辑（happy path）
    results = []
    for image in data["images"]:
        result = process_image(image)
        results.append(result)
    
    return {"results": results, "count": len(results)}

# 3. 使用默认参数而非None检查
# ✅ 好的示例
def search(query: str, limit: int = 10, filters: Dict = None) -> List:
    filters = filters or {}
    # 继续处理
    
# ❌ 不好的示例
def search(query: str, limit: Optional[int], filters: Optional[Dict]) -> List:
    if limit is None:
        limit = 10
    if filters is None:
        filters = {}
    # 继续处理

# 4. 参数验证和错误处理
def divide(a: float, b: float) -> float:
    """安全的除法操作"""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Arguments must be numbers")
    
    if b == 0:
        raise ValueError("Cannot divide by zero")
    
    return a / b

# 5. 使用上下文管理器
from contextlib import contextmanager

@contextmanager
def temporary_directory():
    """创建临时目录的上下文管理器"""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir)

# 使用
with temporary_directory() as temp_dir:
    # 在临时目录中操作
    process_files(temp_dir)
```

### 异步编程规范
```python
import asyncio
from typing import List, Dict
import aiohttp
import aiofiles

# 异步函数定义
async def fetch_data(url: str) -> Dict:
    """异步获取数据"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# 异步文件操作
async def read_file_async(path: Path) -> str:
    """异步读取文件"""
    async with aiofiles.open(path, mode='r') as f:
        content = await f.read()
    return content

# 并发执行多个异步任务
async def process_batch_async(urls: List[str]) -> List[Dict]:
    """并发处理多个URL"""
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 过滤掉异常
    valid_results = []
    for result in results:
        if not isinstance(result, Exception):
            valid_results.append(result)
        else:
            logger.error(f"Failed to fetch: {result}")
    
    return valid_results

# 异步生成器
async def read_large_file_chunks(path: Path, chunk_size: int = 1024):
    """异步读取大文件的生成器"""
    async with aiofiles.open(path, mode='rb') as f:
        while True:
            chunk = await f.read(chunk_size)
            if not chunk:
                break
            yield chunk

# 异步上下文管理器
class AsyncDatabaseConnection:
    async def __aenter__(self):
        self.conn = await asyncpg.connect('postgresql://...')
        return self.conn
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.conn.close()

# 使用异步上下文管理器
async with AsyncDatabaseConnection() as conn:
    result = await conn.fetch('SELECT * FROM photos')
```

### 错误处理规范
```python
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

# 1. 使用自定义异常
class VibePhotosError(Exception):
    """基础异常类"""
    pass

class ImageNotFoundError(VibePhotosError):
    """图像未找到异常"""
    pass

class ProcessingError(VibePhotosError):
    """处理错误异常"""
    pass

# 2. 明确的错误处理
def safe_process_image(image_path: Path) -> Optional[Dict]:
    """安全的图像处理，包含完整错误处理"""
    try:
        # 检查文件存在
        if not image_path.exists():
            raise ImageNotFoundError(f"Image not found: {image_path}")
        
        # 检查文件大小
        file_size = image_path.stat().st_size
        if file_size > 100 * 1024 * 1024:  # 100MB
            raise ProcessingError("File too large")
        
        # 处理图像
        result = process_image(image_path)
        
        return result
        
    except ImageNotFoundError as e:
        logger.warning(f"Image not found: {e}")
        return None
        
    except ProcessingError as e:
        logger.error(f"Processing failed: {e}")
        return None
        
    except Exception as e:
        logger.exception(f"Unexpected error processing {image_path}")
        raise  # 重新抛出未预期的异常

# 3. 使用Result类型模式
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar('T')

@dataclass
class Result(Generic[T]):
    """结果包装类，包含成功值或错误"""
    value: Optional[T] = None
    error: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        return self.error is None
    
    @property
    def is_failure(self) -> bool:
        return self.error is not None
    
    @classmethod
    def success(cls, value: T) -> 'Result[T]':
        return cls(value=value)
    
    @classmethod
    def failure(cls, error: str) -> 'Result[T]':
        return cls(error=error)

def divide_safe(a: float, b: float) -> Result[float]:
    """安全除法，返回Result"""
    if b == 0:
        return Result.failure("Division by zero")
    return Result.success(a / b)

# 使用Result
result = divide_safe(10, 2)
if result.is_success:
    print(f"Result: {result.value}")
else:
    print(f"Error: {result.error}")

# 4. 重试机制
from functools import wraps
import time

def retry(max_attempts: int = 3, delay: float = 1.0):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay * (2 ** attempt))  # 指数退避
            return None
        return wrapper
    return decorator

@retry(max_attempts=3, delay=1.0)
def unreliable_operation():
    """可能失败的操作"""
    import random
    if random.random() < 0.7:
        raise ConnectionError("Network error")
    return "Success"
```

### 日志规范
```python
import logging
from functools import wraps
import time

# 配置日志
def setup_logging(level: str = "INFO"):
    """配置日志系统"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )

# 获取logger
logger = logging.getLogger(__name__)

# 日志级别使用
def process_data(data: Dict) -> Dict:
    logger.debug(f"Processing data with {len(data)} items")
    
    try:
        # 信息级日志
        logger.info("Starting data processing")
        
        # 处理逻辑
        result = transform_data(data)
        
        # 成功日志
        logger.info(f"Successfully processed {len(result)} items")
        
        return result
        
    except ValueError as e:
        # 警告级日志
        logger.warning(f"Invalid data format: {e}")
        return {}
        
    except Exception as e:
        # 错误级日志
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise

# 性能日志装饰器
def log_performance(func):
    """记录函数执行时间的装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        logger.debug(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
            return result
            
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
            raise
    
    return wrapper

@log_performance
def expensive_operation():
    """耗时操作"""
    time.sleep(2)
    return "Done"

# 结构化日志
def log_structured(event: str, **kwargs):
    """结构化日志记录"""
    import json
    log_entry = {
        "event": event,
        "timestamp": time.time(),
        **kwargs
    }
    logger.info(json.dumps(log_entry))

# 使用结构化日志
log_structured(
    "image_processed",
    image_path="/path/to/image.jpg",
    category="electronic",
    confidence=0.95,
    processing_time=0.234
)
```

### 测试规范
```python
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

# 1. 测试文件组织
# test_<module_name>.py 对应 <module_name>.py

# 2. 测试类和函数命名
class TestImageDetector:
    """测试ImageDetector类"""
    
    def test_initialization(self):
        """测试初始化"""
        detector = ImageDetector()
        assert detector is not None
    
    def test_detect_valid_image(self):
        """测试检测有效图像"""
        detector = ImageDetector()
        result = detector.detect("test.jpg")
        assert result["status"] == "success"
    
    def test_detect_invalid_image(self):
        """测试检测无效图像"""
        detector = ImageDetector()
        with pytest.raises(ImageNotFoundError):
            detector.detect("nonexistent.jpg")

# 3. 使用fixtures
@pytest.fixture
def temp_image_file():
    """创建临时图像文件的fixture"""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        # 创建测试图像
        f.write(b"fake image data")
        temp_path = Path(f.name)
    
    yield temp_path
    
    # 清理
    temp_path.unlink()

@pytest.fixture
def mock_detector():
    """创建模拟检测器的fixture"""
    detector = Mock(spec=ImageDetector)
    detector.detect.return_value = {
        "category": "electronic",
        "confidence": 0.95
    }
    return detector

# 4. 参数化测试
@pytest.mark.parametrize("input_value,expected", [
    ("", False),
    ("valid", True),
    (None, False),
    ("test@example.com", True),
])
def test_validate_input(input_value, expected):
    """参数化测试验证输入"""
    result = validate_input(input_value)
    assert result == expected

# 5. 测试异步函数
@pytest.mark.asyncio
async def test_async_fetch():
    """测试异步获取函数"""
    async with aiohttp.ClientSession() as session:
        data = await fetch_data("https://api.example.com/data")
        assert data is not None

# 6. 测试异常
def test_division_by_zero():
    """测试除零异常"""
    with pytest.raises(ZeroDivisionError):
        result = divide(10, 0)

# 7. Mock外部依赖
@patch('src.core.detector.load_model')
def test_detector_with_mock_model(mock_load_model):
    """使用mock测试检测器"""
    # 设置mock返回值
    mock_model = MagicMock()
    mock_model.predict.return_value = {"category": "test"}
    mock_load_model.return_value = mock_model
    
    # 测试
    detector = ImageDetector()
    result = detector.detect("test.jpg")
    
    # 验证
    assert result["category"] == "test"
    mock_load_model.assert_called_once()
    mock_model.predict.assert_called_once()

# 8. 性能测试
@pytest.mark.benchmark
def test_performance(benchmark):
    """性能基准测试"""
    def process():
        return expensive_operation()
    
    result = benchmark(process)
    assert result is not None
    
    # 检查性能指标
    assert benchmark.stats["mean"] < 1.0  # 平均时间小于1秒

# 9. 集成测试
@pytest.mark.integration
class TestAPIIntegration:
    """API集成测试"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """测试设置"""
        self.client = TestClient(app)
        
    def test_full_workflow(self):
        """测试完整工作流"""
        # 1. 上传图像
        response = self.client.post("/upload", files={"file": ("test.jpg", b"data")})
        assert response.status_code == 200
        photo_id = response.json()["id"]
        
        # 2. 获取检测结果
        response = self.client.get(f"/photos/{photo_id}")
        assert response.status_code == 200
        assert response.json()["category"] is not None
        
        # 3. 搜索
        response = self.client.get("/search?q=test")
        assert response.status_code == 200
        assert len(response.json()["results"]) > 0
```

### FastAPI规范
```python
from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime
import asyncio

# 创建应用
app = FastAPI(
    title="Vibe Photos API",
    description="AI智能照片管理系统API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic模型
class PhotoBase(BaseModel):
    """照片基础模型"""
    path: str = Field(..., description="照片路径")
    category: Optional[str] = Field(None, description="分类")
    
    class Config:
        schema_extra = {
            "example": {
                "path": "/photos/IMG_001.jpg",
                "category": "electronic"
            }
        }

class PhotoCreate(PhotoBase):
    """创建照片模型"""
    pass

class PhotoResponse(PhotoBase):
    """照片响应模型"""
    id: int
    confidence: float
    created_at: datetime
    
    class Config:
        orm_mode = True

class SearchQuery(BaseModel):
    """搜索查询模型"""
    q: str = Field(..., min_length=1, max_length=100, description="搜索关键词")
    limit: int = Field(20, ge=1, le=100, description="结果限制")
    category: Optional[str] = Field(None, description="分类过滤")
    
    @validator('q')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

# 依赖注入
async def get_db_session():
    """获取数据库会话"""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """获取当前用户"""
    # 验证token
    user = verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

# 路由定义
@app.get("/", response_model=Dict[str, str])
async def root():
    """API根路径"""
    return {
        "name": "Vibe Photos API",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/photos", response_model=PhotoResponse)
async def create_photo(
    photo: PhotoCreate,
    db: Session = Depends(get_db_session),
    current_user: User = Depends(get_current_user)
):
    """创建照片记录"""
    db_photo = Photo(**photo.dict())
    db.add(db_photo)
    db.commit()
    db.refresh(db_photo)
    return db_photo

@app.get("/photos/{photo_id}", response_model=PhotoResponse)
async def get_photo(
    photo_id: int = Path(..., gt=0, description="照片ID"),
    db: Session = Depends(get_db_session)
):
    """获取单张照片"""
    photo = db.query(Photo).filter(Photo.id == photo_id).first()
    if not photo:
        raise HTTPException(status_code=404, detail="Photo not found")
    return photo

@app.get("/search", response_model=List[PhotoResponse])
async def search_photos(
    query: SearchQuery = Depends(),
    db: Session = Depends(get_db_session)
):
    """搜索照片"""
    photos = db.query(Photo).filter(
        Photo.caption.contains(query.q)
    ).limit(query.limit).all()
    return photos

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks
):
    """上传文件"""
    # 验证文件类型
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # 保存文件
    file_path = save_uploaded_file(file)
    
    # 添加后台任务
    background_tasks.add_task(process_image, file_path)
    
    return {"filename": file.filename, "status": "processing"}

# 异常处理
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """处理值错误"""
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)}
    )

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """处理404错误"""
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found"}
    )

# 中间件
@app.middleware("http")
async def log_requests(request, call_next):
    """记录请求日志"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {process_time:.2f}s")
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

# 启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("Application starting up...")
    # 初始化数据库连接
    # 加载模型
    # 预热缓存

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("Application shutting down...")
    # 关闭数据库连接
    # 清理资源
```

## 📋 代码审查清单

### 功能性检查
- [ ] 代码是否实现了预期功能？
- [ ] 边界情况是否处理？
- [ ] 错误情况是否妥善处理？
- [ ] 是否有未使用的代码？

### 代码质量检查
- [ ] 命名是否清晰描述性？
- [ ] 函数是否遵循单一职责原则？
- [ ] 是否有重复代码可以抽取？
- [ ] 复杂度是否可以降低？

### 类型和文档检查
- [ ] 类型注解是否完整？
- [ ] 文档字符串是否清晰？
- [ ] 注释是否必要且有用？
- [ ] README是否更新？

### 测试检查
- [ ] 是否有对应的测试？
- [ ] 测试覆盖率是否足够？
- [ ] 测试是否可以独立运行？
- [ ] 是否有集成测试？

### 性能和安全检查
- [ ] 是否有性能瓶颈？
- [ ] 是否有内存泄漏风险？
- [ ] 是否有SQL注入风险？
- [ ] 敏感信息是否安全处理？

## 🚀 性能优化指南

### 代码层优化
```python
# 1. 使用生成器避免内存占用
# ❌ 不好 - 一次性加载所有数据
def load_all_images(directory: Path) -> List[Image]:
    images = []
    for img_path in directory.glob("*.jpg"):
        images.append(load_image(img_path))
    return images

# ✅ 好 - 使用生成器延迟加载
def load_images_generator(directory: Path):
    for img_path in directory.glob("*.jpg"):
        yield load_image(img_path)

# 2. 缓存重复计算
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(param: str) -> Dict:
    """缓存昂贵的计算结果"""
    # 复杂计算
    return result

# 3. 批处理优化
def process_batch(items: List, batch_size: int = 32):
    """批处理以提高效率"""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        # 批量处理
        yield process_items(batch)

# 4. 使用numpy向量化操作
import numpy as np

# ❌ 不好 - Python循环
def compute_similarity_slow(vec1: List, vec2: List) -> float:
    result = 0
    for i in range(len(vec1)):
        result += vec1[i] * vec2[i]
    return result

# ✅ 好 - NumPy向量化
def compute_similarity_fast(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2)

# 5. 连接池复用
from sqlalchemy.pool import QueuePool

engine = create_engine(
    "postgresql://...",
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=0,
    pool_pre_ping=True
)
```

### 异步优化
```python
# 并发处理多个任务
async def process_images_concurrent(image_paths: List[Path]):
    """并发处理多个图像"""
    tasks = []
    for path in image_paths:
        task = asyncio.create_task(process_image_async(path))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results

# 限制并发数
async def process_with_semaphore(items: List, max_concurrent: int = 10):
    """使用信号量限制并发数"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_item(item):
        async with semaphore:
            return await process_async(item)
    
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks)
```

## 🔍 调试技巧

### 使用调试工具
```python
# 1. 使用pdb调试
import pdb

def complex_function(data):
    processed = preprocess(data)
    pdb.set_trace()  # 断点
    result = transform(processed)
    return result

# 2. 使用logging调试
import logging
logging.basicConfig(level=logging.DEBUG)

def debug_function(data):
    logger.debug(f"Input data: {data}")
    result = process(data)
    logger.debug(f"Output result: {result}")
    return result

# 3. 使用装饰器追踪
def trace(func):
    """追踪函数调用的装饰器"""
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

@trace
def calculate(a, b):
    return a + b

# 4. 性能分析
import cProfile
import pstats

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 要分析的代码
    expensive_operation()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

## 📝 Git提交规范

### 提交消息格式
```
<type>(<scope>): <subject>

<body>

<footer>
```

### 类型（type）
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建/工具链相关

### 示例
```bash
# 好的提交消息
git commit -m "feat(detector): add SigLIP model support for multi-language classification"
git commit -m "fix(api): handle null values in search query"
git commit -m "docs(readme): update installation instructions"
git commit -m "perf(processor): optimize batch processing with parallel execution"

# 不好的提交消息
git commit -m "update"
git commit -m "fix bug"
git commit -m "changes"
```

## 🎯 代码质量目标

### 必须达到的指标
- **测试覆盖率**: ≥ 80%
- **代码复杂度**: 圈复杂度 < 10
- **函数长度**: < 50行
- **类长度**: < 300行
- **文件长度**: < 500行
- **响应时间**: P95 < 500ms
- **错误率**: < 1%

### 代码质量工具配置
```yaml
# .ruff.toml
line-length = 100
target-version = "py312"

[lint]
select = ["E", "F", "I", "N", "W", "UP", "ASYNC", "B", "A", "C4", "DTZ", "T10", "EM", "ISC", "ICN", "T20", "Q", "RET", "SIM", "TID", "ARG", "ERA", "PD", "PGH", "PL", "TRY", "NPY", "RUF"]
ignore = ["E501"]

# pytest.ini
[tool.pytest.ini_options]
minversion = "6.0"
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-ra -q --strict-markers --cov=src --cov-report=term-missing"

# mypy.ini
[mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_unimported = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
check_untyped_defs = true
```

---

**文档版本**: 1.0.0
**最后更新**: 2024-11-12
**适用于**: Python 3.12 + FastAPI + uv
**目标**: 保证代码质量和一致性
