# Contributing to ArUco Pose Estimation Toolkit

Thank you for considering contributing to this project! 

## 🚀 Quick Start for Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ArUco_Pose_Estimation.git
   cd ArUco_Pose_Estimation
   ```
3. **Install in development mode**:
   ```bash
   pip install -e .
   ```
4. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/my-new-feature
   ```

## 📋 Development Guidelines

### Code Style
- Follow PEP 8 conventions
- Use type hints where appropriate
- Add docstrings to all public functions/classes
- Keep functions focused and modular

### Project Structure
- **Core functionality** → `src/core/`
- **3D models** → `src/models/`
- **Tracking algorithms** → `src/tracking/`
- **Utilities** → `src/utils/`
- **Examples** → `examples/` (organized by complexity)
- **CLI tools** → `tools/`
- **Tests** → `tests/`

### Adding New Features

1. **Create appropriate files** in the right directory
2. **Update `__init__.py`** files to export new classes/functions
3. **Add examples** showing usage
4. **Write tests** for new functionality
5. **Update documentation** in `docs/`

### Testing

Run tests before submitting:
```bash
pytest tests/
```

Add new tests for your features in `tests/test_*.py`

## 📝 Documentation

- Update relevant README files when adding features
- Add docstrings to all public APIs
- Include usage examples in docstrings
- Update `docs/` files if architecture changes

## 🐛 Reporting Bugs

When reporting bugs, please include:
- Python version
- OpenCV version
- Operating system
- Minimal code to reproduce
- Expected vs actual behavior
- Error messages/stack traces

## 💡 Suggesting Features

Feature requests are welcome! Please:
- Check if it already exists in issues
- Explain the use case
- Provide example usage if possible

## 🔀 Pull Request Process

1. **Update documentation** for any changed functionality
2. **Add tests** for new features
3. **Ensure tests pass**: `pytest tests/`
4. **Update CHANGELOG** (if applicable)
5. **Submit PR** with clear description of changes

### PR Checklist
- [ ] Code follows project style
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Examples added/updated if needed
- [ ] No breaking changes (or clearly documented)

## 📦 Release Process

Maintainers will:
1. Review and merge PRs
2. Update version in `setup.py`
3. Update CHANGELOG
4. Create GitHub release
5. Tag version

## 🤝 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn

## 📧 Questions?

- Open an issue for technical questions
- Email maintainer for other inquiries: jshulgac@andrew.cmu.edu

---

Thank you for contributing! 🎉
