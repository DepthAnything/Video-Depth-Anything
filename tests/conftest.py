"""Shared pytest fixtures and configuration for all tests."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import torch
import numpy as np
from PIL import Image


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory that is cleaned up after the test."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image_path(temp_dir: Path) -> Path:
    """Create a sample RGB image for testing."""
    img_path = temp_dir / "test_image.png"
    # Create a simple 100x100 RGB image
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_video_path(temp_dir: Path) -> Path:
    """Create a sample video file path for testing (mock)."""
    # Note: This is just a path fixture for testing file handling
    # Actual video creation would require imageio-ffmpeg
    video_path = temp_dir / "test_video.mp4"
    video_path.touch()  # Create empty file for path testing
    return video_path


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """Create a sample torch tensor for testing."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_depth_map() -> np.ndarray:
    """Create a sample depth map for testing."""
    return np.random.rand(480, 640).astype(np.float32)


@pytest.fixture
def mock_model_config() -> dict:
    """Mock configuration for model testing."""
    return {
        "encoder": "dinov2",
        "decoder": "dpt",
        "input_size": [224, 224],
        "output_size": [480, 640],
        "pretrained": False,
        "temporal": True,
        "motion_module": {
            "enabled": True,
            "num_frames": 8
        }
    }


@pytest.fixture
def mock_dataset_config() -> dict:
    """Mock configuration for dataset testing."""
    return {
        "name": "test_dataset",
        "root_dir": "/tmp/test_data",
        "batch_size": 4,
        "num_workers": 2,
        "shuffle": True,
        "transform": {
            "resize": [224, 224],
            "normalize": True
        }
    }


@pytest.fixture
def environment_setup(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent.parent))
    

@pytest.fixture(autouse=True)
def torch_deterministic():
    """Make torch operations deterministic for testing."""
    torch.manual_seed(42)
    np.random.seed(42)
    # Store original settings
    old_deterministic = torch.backends.cudnn.deterministic
    old_benchmark = torch.backends.cudnn.benchmark
    
    # Set deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    yield
    
    # Restore original settings
    torch.backends.cudnn.deterministic = old_deterministic
    torch.backends.cudnn.benchmark = old_benchmark


@pytest.fixture
def gpu_available() -> bool:
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture
def skip_if_no_gpu(gpu_available):
    """Skip test if GPU is not available."""
    if not gpu_available:
        pytest.skip("GPU not available")


class MockResponse:
    """Mock HTTP response for testing."""
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code
        
    def raise_for_status(self):
        if self.status_code != 200:
            raise Exception(f"HTTP {self.status_code}")


@pytest.fixture
def mock_http_response():
    """Factory for creating mock HTTP responses."""
    def _create_response(content: bytes = b"test", status_code: int = 200):
        return MockResponse(content, status_code)
    return _create_response


# Markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")