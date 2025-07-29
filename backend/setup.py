#!/usr/bin/env python3
"""
Setup script for Cortex Python SDK
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cortex-memory",
    version="2.0.0",
    author="Cortex Team",
    author_email="team@cortex.ai",
    description="Enterprise-Grade Context-Aware AI System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VaishakhVipin/cortex-memory",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0",
            "pydantic>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cortex=cortex.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cortex": ["*.md", "*.txt"],
    },
    keywords="ai, memory, context, semantic, embeddings, llm, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/VaishakhVipin/cortex-memory/issues",
        "Source": "https://github.com/VaishakhVipin/cortex-memory",
        "Documentation": "https://cortex-memory.readthedocs.io/",
    },
)