from __future__ import annotations

import tomllib
from pathlib import Path


def test_cuda_dependencies_are_linux_only_optional_extra():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]
    assert not any("nvidia-" in dependency for dependency in dependencies)

    cuda_extra = pyproject["project"]["optional-dependencies"]["cuda"]
    assert any("nvidia-cublas-cu12" in dependency for dependency in cuda_extra)
    assert any("nvidia-cudnn-cu12" in dependency for dependency in cuda_extra)
    assert all("sys_platform == 'linux'" in dependency for dependency in cuda_extra)
    assert all("platform_machine == 'x86_64'" in dependency for dependency in cuda_extra)
