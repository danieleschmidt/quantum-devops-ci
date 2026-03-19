"""Tests for CITemplate."""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from quantum_devops.ci_template import CITemplate


class TestCITemplate:
    def test_generate_returns_yaml_string(self):
        t = CITemplate(repo_name="test-repo")
        yaml = t.generate()
        assert isinstance(yaml, str)
        assert len(yaml) > 100

    def test_yaml_contains_repo_name(self):
        t = CITemplate(repo_name="my-quantum-repo")
        yaml = t.generate()
        assert "my-quantum-repo" in yaml

    def test_yaml_contains_python_versions(self):
        t = CITemplate(python_versions=["3.10", "3.11"])
        yaml = t.generate()
        assert "3.10" in yaml
        assert "3.11" in yaml

    def test_yaml_contains_fidelity_threshold(self):
        t = CITemplate(fidelity_threshold=0.75)
        yaml = t.generate()
        assert "0.75" in yaml

    def test_yaml_contains_noise_levels(self):
        t = CITemplate(noise_levels=[0.0, 0.05])
        yaml = t.generate()
        assert "0.0" in yaml
        assert "0.05" in yaml

    def test_yaml_contains_shots(self):
        t = CITemplate(shots=500)
        yaml = t.generate()
        assert "500" in yaml

    def test_yaml_valid_structure(self):
        """Check YAML has the expected workflow sections."""
        t = CITemplate()
        yaml = t.generate()
        assert "name:" in yaml
        assert "on:" in yaml
        assert "jobs:" in yaml
        assert "steps:" in yaml
        assert "actions/checkout" in yaml
        assert "actions/setup-python" in yaml

    def test_yaml_has_noise_test_job(self):
        t = CITemplate()
        yaml = t.generate()
        assert "noise-aware-tests" in yaml
        assert "noise-aware" in yaml.lower() or "Noise-Aware" in yaml

    def test_yaml_has_resource_job(self):
        t = CITemplate(enable_resource_estimation=True)
        yaml = t.generate()
        assert "resource-estimation" in yaml

    def test_no_resource_estimation(self):
        t = CITemplate(enable_resource_estimation=False)
        yaml = t.generate()
        # Resource step content should not be present
        assert "ResourceEstimator" not in yaml

    def test_branches_in_yaml(self):
        t = CITemplate(on_push_branches=["main", "feature/test"])
        yaml = t.generate()
        assert "main" in yaml
        assert "feature/test" in yaml

    def test_save_creates_file(self, tmp_path):
        t = CITemplate(repo_name="test")
        path = str(tmp_path / ".github" / "workflows" / "quantum-ci.yml")
        saved = t.save(path)
        assert saved == path
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "test" in content

    def test_minimal_template(self):
        yaml = CITemplate.minimal(repo_name="quick-project")
        assert "quantum-devops-ci" in yaml
        assert "pytest" in yaml
