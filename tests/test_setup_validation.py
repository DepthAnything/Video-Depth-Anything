"""Validation tests to ensure the testing infrastructure is properly set up."""

import pytest
import sys
from pathlib import Path


class TestSetupValidation:
    """Validate that the testing infrastructure is correctly configured."""
    
    @pytest.mark.unit
    def test_pytest_installed(self):
        """Verify pytest is installed and importable."""
        import pytest
        assert pytest.__version__
        
    @pytest.mark.unit
    def test_pytest_cov_installed(self):
        """Verify pytest-cov is installed."""
        import pytest_cov
        assert pytest_cov
        
    @pytest.mark.unit
    def test_pytest_mock_installed(self):
        """Verify pytest-mock is installed."""
        import pytest_mock
        assert pytest_mock
        
    @pytest.mark.unit
    def test_project_structure(self):
        """Verify the project structure is set up correctly."""
        project_root = Path(__file__).parent.parent
        
        # Check main directories exist
        assert (project_root / "video_depth_anything").exists()
        assert (project_root / "metric_depth").exists()
        assert (project_root / "benchmark").exists()
        assert (project_root / "utils").exists()
        assert (project_root / "tests").exists()
        
        # Check test structure
        assert (project_root / "tests" / "__init__.py").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
        assert (project_root / "tests" / "conftest.py").exists()
        
    @pytest.mark.unit
    def test_conftest_fixtures(self, temp_dir, sample_tensor, mock_model_config):
        """Test that conftest fixtures are available and working."""
        # Test temp_dir fixture
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test sample_tensor fixture
        assert sample_tensor.shape == (1, 3, 224, 224)
        
        # Test mock_model_config fixture
        assert mock_model_config["encoder"] == "dinov2"
        assert mock_model_config["temporal"] is True
        
    @pytest.mark.unit
    def test_markers_configured(self, request):
        """Verify custom markers are properly configured."""
        markers = request.config.getini("markers")
        marker_names = [m.split(":")[0].strip() for m in markers]
        
        assert "unit" in marker_names
        assert "integration" in marker_names
        assert "slow" in marker_names
        
    @pytest.mark.unit
    def test_coverage_configuration(self):
        """Verify coverage is configured correctly."""
        from pathlib import Path
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists()
        
        content = pyproject_path.read_text()
        assert "[tool.coverage.run]" in content
        assert "[tool.coverage.report]" in content
        assert "cov-fail-under=80" in content
        
    @pytest.mark.unit
    def test_imports_work(self):
        """Test that project modules can be imported."""
        # These imports should work if PYTHONPATH is set correctly
        try:
            import video_depth_anything
            import metric_depth
            import benchmark
            import utils
        except ImportError as e:
            pytest.fail(f"Failed to import project modules: {e}")
            
    @pytest.mark.unit
    def test_torch_deterministic(self):
        """Verify torch deterministic settings are applied."""
        import torch
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        
    @pytest.mark.integration
    def test_sample_integration(self, temp_dir):
        """A simple integration test to verify the marker works."""
        test_file = temp_dir / "integration_test.txt"
        test_file.write_text("Integration test")
        assert test_file.read_text() == "Integration test"
        
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works (this is not actually slow)."""
        import time
        start = time.time()
        # Simulate work
        result = sum(range(1000))
        duration = time.time() - start
        assert result == 499500
        assert duration < 1.0  # Not actually slow