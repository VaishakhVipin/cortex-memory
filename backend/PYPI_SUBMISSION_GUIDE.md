# 📦 PyPI Submission Guide - Cortex Memory

## 🎯 **Overview**

This guide walks you through the complete process of submitting the Cortex Memory package to PyPI (Python Package Index).

## 📋 **Prerequisites**

### **1. PyPI Account**
- Create account at [PyPI](https://pypi.org/account/register/)
- Create account at [Test PyPI](https://test.pypi.org/account/register/) (for testing)

### **2. Required Tools**
```bash
# Install build tools
pip install build twine

# Install development dependencies
pip install -e .[dev]
```

### **3. Package Preparation**
Ensure your package is ready:
- ✅ All files are in `cortex_memory/` directory
- ✅ `pyproject.toml` is properly configured
- ✅ Tests pass
- ✅ Documentation is complete

## 🔧 **Package Configuration**

### **Current pyproject.toml**
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cortex-memory"
version = "2.0.0"
description = "🧠 The Smart Context Layer for Prompt Chains in LLMs - Enterprise-grade context-aware AI system with semantic understanding and self-evolving memory"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Cortex Team", email = "team@cortex-memory.com"}
]
maintainers = [
    {name = "Cortex Team", email = "team@cortex-memory.com"}
]
keywords = [
    "ai", "memory", "context", "semantic", "embeddings", "llm", "prompt-chains",
    "machine-learning", "nlp", "artificial-intelligence", "context-aware"
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Text Processing :: Linguistic",
    "Typing :: Typed",
]
requires-python = ">=3.8"
dependencies = [
    "redis>=4.0.0",
    "requests>=2.25.0",
    "python-dotenv>=0.19.0",
    "fastapi>=0.68.0",
    "uvicorn[standard]>=0.15.0",
    "pydantic>=1.8.0",
    "numpy>=1.21.0",
    "sentence-transformers>=2.2.0",
    "urllib3>=1.26.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=6.0.0",
    "pytest-asyncio>=0.18.0",
    "pytest-cov>=2.12.0",
    "black>=21.0.0",
    "isort>=5.9.0",
    "flake8>=3.9.0",
    "mypy>=0.910",
    "pre-commit>=2.15.0",
]
api = [
    "fastapi>=0.68.0",
    "uvicorn[standard]>=0.15.0",
    "python-multipart>=0.0.5",
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
    "sqlalchemy>=1.4.0",
    "alembic>=1.7.0",
    "psycopg2-binary>=2.9.0",
]
monitoring = [
    "prometheus-client>=0.12.0",
    "structlog>=21.1.0",
    "sentry-sdk[fastapi]>=1.5.0",
]

[project.urls]
Homepage = "https://github.com/cortex-memory/cortex-memory"
Documentation = "https://docs.cortex-memory.com"
Repository = "https://github.com/cortex-memory/cortex-memory"
"Bug Tracker" = "https://github.com/cortex-memory/cortex-memory/issues"
"Source Code" = "https://github.com/cortex-memory/cortex-memory"
"Download" = "https://pypi.org/project/cortex-memory/#files"
"Changelog" = "https://github.com/cortex-memory/cortex-memory/blob/main/CHANGELOG.md"

[project.scripts]
cortex-memory = "cortex_memory.cli:main"

[project.entry-points."cortex_memory.plugins"]
default = "cortex_memory.core:DefaultPlugin"
```

## 🧪 **Pre-Submission Testing**

### **1. Run All Tests**
```bash
# Navigate to backend directory
cd backend

# Run tests
pytest

# Run with coverage
pytest --cov=cortex_memory --cov-report=html

# Run linting
black cortex_memory/
isort cortex_memory/
flake8 cortex_memory/
mypy cortex_memory/
```

### **2. Test Package Build**
```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build package
python -m build

# Check build artifacts
ls dist/
# Should show:
# cortex_memory-2.0.0.tar.gz
# cortex_memory-2.0.0-py3-none-any.whl
```

### **3. Test Package Installation**
```bash
# Test installation from wheel
pip install dist/cortex_memory-2.0.0-py3-none-any.whl

# Test import
python -c "from cortex_memory import create_client; print('✅ Package works!')"

# Uninstall
pip uninstall cortex-memory -y
```

### **4. Test on Test PyPI**
```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ cortex-memory

# Test functionality
python -c "from cortex_memory import create_client; print('✅ Test PyPI works!')"

# Uninstall
pip uninstall cortex-memory -y
```

## 📦 **Building for Production**

### **1. Update Version**
```bash
# Edit pyproject.toml to update version
# Change version = "2.0.0" to version = "2.0.1" (or next version)
```

### **2. Clean Build**
```bash
# Remove old builds
rm -rf build/ dist/ *.egg-info/

# Build fresh package
python -m build
```

### **3. Verify Build**
```bash
# Check package contents
tar -tzf dist/cortex_memory-2.0.0.tar.gz | head -20

# Should show:
# cortex_memory-2.0.0/
# cortex_memory-2.0.0/cortex_memory/
# cortex_memory-2.0.0/cortex_memory/__init__.py
# cortex_memory-2.0.0/cortex_memory/core.py
# cortex_memory-2.0.0/cortex_memory/client.py
# ...
```

## 🚀 **Submitting to PyPI**

### **1. Upload to PyPI**
```bash
# Upload to production PyPI
twine upload dist/*

# You'll be prompted for:
# Username: your_pypi_username
# Password: your_pypi_password
```

### **2. Verify Upload**
```bash
# Check package on PyPI
# Visit: https://pypi.org/project/cortex-memory/

# Test installation
pip install cortex-memory

# Test functionality
python -c "from cortex_memory import create_client; print('✅ PyPI package works!')"
```

## 📋 **Post-Submission Checklist**

### **1. PyPI Package Page**
- ✅ Package name is correct: `cortex-memory`
- ✅ Description is clear and compelling
- ✅ README.md is displayed properly
- ✅ All classifiers are correct
- ✅ Dependencies are listed correctly
- ✅ Project URLs are working

### **2. Installation Test**
```bash
# Test fresh installation
pip install cortex-memory

# Test import
python -c "import cortex_memory; print(cortex_memory.__version__)"

# Test basic functionality
python -c "from cortex_memory import create_client; print('✅ Success!')"
```

### **3. Documentation**
- ✅ Update GitHub README with PyPI installation instructions
- ✅ Update documentation links
- ✅ Create release notes

## 🔄 **Updating the Package**

### **1. Version Bumping**
```bash
# Update version in pyproject.toml
# version = "2.0.0" → version = "2.0.1"
```

### **2. Build and Upload**
```bash
# Clean and build
rm -rf build/ dist/ *.egg-info/
python -m build

# Upload new version
twine upload dist/*
```

### **3. Verify Update**
```bash
# Check new version is available
pip install --upgrade cortex-memory
python -c "import cortex_memory; print(cortex_memory.__version__)"
```

## 🛠️ **Troubleshooting**

### **Common Issues**

#### **1. Package Name Already Taken**
```
HTTPError: 400 Client Error: File already exists.
```
**Solution:**
- Check if package name is available: https://pypi.org/project/cortex-memory/
- Use a different name or contact existing maintainer

#### **2. Authentication Error**
```
HTTPError: 401 Client Error: Unauthorized
```
**Solution:**
- Verify PyPI credentials
- Use API token instead of password
- Check if account is activated

#### **3. Build Errors**
```
error: invalid command 'bdist_wheel'
```
**Solution:**
```bash
pip install wheel
python -m build
```

#### **4. Import Errors After Installation**
```
ModuleNotFoundError: No module named 'cortex_memory'
```
**Solution:**
- Check package structure
- Verify `__init__.py` files exist
- Check `pyproject.toml` configuration

## 📊 **Package Analytics**

### **1. PyPI Statistics**
- Visit: https://pypi.org/project/cortex-memory/#statistics
- Monitor download statistics
- Track version adoption

### **2. Installation Analytics**
```bash
# Check package info
pip show cortex-memory

# List installed files
pip show -f cortex-memory
```

## 🎯 **Best Practices**

### **1. Version Management**
- Use semantic versioning (MAJOR.MINOR.PATCH)
- Document breaking changes
- Maintain changelog

### **2. Quality Assurance**
- Always test on Test PyPI first
- Run full test suite before release
- Check package contents before upload

### **3. Documentation**
- Keep README.md updated
- Include usage examples
- Document breaking changes

### **4. Security**
- Use API tokens instead of passwords
- Regularly update dependencies
- Monitor for security vulnerabilities

## 🚀 **Next Steps After PyPI**

### **1. Announcement**
- Create GitHub release
- Post on social media
- Update documentation

### **2. Monitoring**
- Monitor PyPI statistics
- Track user feedback
- Monitor for issues

### **3. Maintenance**
- Regular dependency updates
- Bug fixes and improvements
- Feature additions

---

## 📞 **Support**

### **PyPI Help**
- [PyPI Help](https://pypi.org/help/)
- [Packaging Guide](https://packaging.python.org/)
- [Twine Documentation](https://twine.readthedocs.io/)

### **Package Issues**
- Check PyPI package page for issues
- Monitor GitHub issues
- Respond to user feedback

**Your Cortex Memory package is now ready for PyPI submission!** 🎉 