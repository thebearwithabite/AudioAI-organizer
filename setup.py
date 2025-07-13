from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="audioai-organizer",
    version="1.0.16
    author="RT Max",
    author_email="rt@papersthatdream.com",
    description="Intelligent audio library organization with AI-powered analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thebearwithabite/audioai-organizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "librosa>=0.10.0",
        "mutagen>=1.47.0",
        "pandas>=1.5.0",
        "openpyxl>=3.1.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "soundfile>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "jupyter>=1.0.0",
            "tqdm>=4.64.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "audioai=audioai_organizer:main",
        ],
    },
)
