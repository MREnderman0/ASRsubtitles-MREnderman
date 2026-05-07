from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def read_requirements(path: Path) -> list[str]:
    requirements: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


def read_long_description() -> str:
    readme = ROOT / "README.md"
    return readme.read_text(encoding="utf-8") if readme.exists() else ""


setup(
    name="asr-mrenderman",
    version="1.1.0",
    description="Local ASR subtitle generator with LLM boundary planning and review.",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["core", "core.*", "subtitle_rag", "subtitle_rag.*", "scripts"]),
    py_modules=[],
    install_requires=read_requirements(ROOT / "requirements.txt"),
    python_requires=">=3.10,<3.13",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "asr-mrenderman-asr-only=scripts.run_asr_only:main",
        ]
    },
)
