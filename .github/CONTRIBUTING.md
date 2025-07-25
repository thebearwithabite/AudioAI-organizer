# Contributing to AudioAI Organizer

Thank you for your interest in contributing to AudioAI Organizer! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- OpenAI API key for testing AI features
- Audio files for testing (various formats recommended)

### Development Setup
1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/AudioAI-organizer.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install development dependencies: `pip install -e .[dev]`
6. Set up environment variables:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export AUDIOAI_BASE_DIRECTORY="/path/to/test/library"
   ```

## Development Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and single-purpose
- Maximum line length: 127 characters

### Testing
- Write tests for all new features
- Ensure existing tests pass: `pytest tests/`
- Include tests for edge cases and error conditions
- Test with various audio formats (MP3, WAV, FLAC, etc.)
- Mock external API calls in tests

### Documentation
- Update documentation for new features
- Include examples in docstrings
- Update README.md if needed
- Add entries to CHANGELOG.md for significant changes

## Types of Contributions

### Bug Reports
- Use the bug report template
- Include audio file details and error messages
- Provide steps to reproduce
- Test with the latest version

### Feature Requests
- Use the feature request template
- Explain the use case and benefits
- Consider implementation complexity
- Discuss in issues before large changes

### Code Contributions
1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes following the guidelines above
3. Add tests for your changes
4. Ensure all tests pass
5. Update documentation as needed
6. Commit with descriptive messages
7. Push and create a pull request

### Documentation Improvements
- Fix typos and unclear explanations
- Add examples and use cases
- Improve API documentation
- Update setup and troubleshooting guides

## AI and Classification Guidelines

When working on classification features:
- Test with diverse audio content
- Consider edge cases (silence, noise, unusual formats)
- Validate classification accuracy
- Respect API rate limits during testing
- Consider performance implications

## Audio File Considerations

- Test with various formats: MP3, WAV, FLAC, M4A, OGG
- Consider different sample rates and bit depths
- Test with large files (>100MB) and small files (<1MB)
- Handle corrupted or invalid audio files gracefully
- Respect copyright when sharing test files

## Pull Request Process

1. Ensure your PR has a clear description
2. Link to relevant issues
3. Include test results and examples
4. Be responsive to review feedback
5. Squash commits before merging if requested

## Code Review Guidelines

When reviewing code:
- Check for adherence to style guidelines
- Verify test coverage and quality
- Test the changes locally
- Consider performance implications
- Suggest improvements constructively

## Getting Help

- Ask questions in GitHub issues
- Use the question template for usage help
- Check existing documentation first
- Be specific about your setup and problem

## Community Standards

- Be respectful and inclusive
- Help others learn and improve
- Give constructive feedback
- Follow the Code of Conduct

## Recognition

Contributors will be acknowledged in:
- CHANGELOG.md for significant contributions
- GitHub contributors section
- Release notes for major features

Thank you for contributing to AudioAI Organizer!