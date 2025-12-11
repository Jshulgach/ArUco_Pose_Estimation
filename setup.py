from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="aruco-pose-estimation",
    version="2.0.0",
    author="Jonathan Shulgach",
    author_email="jshulgac@andrew.cmu.edu",
    description="A robust toolkit for ArUco marker detection, pose estimation, and custom 3D model tracking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jshulgach/ArUco_Pose_Estimation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.6.0",
        "opencv-contrib-python>=4.6.0",
        "pyyaml>=5.4",
        "packaging>=20.0",
        "click>=8.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
            'mypy>=0.910',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=0.5',
        ],
    },
    entry_points={
        'console_scripts': [
            'aruco-cli=tools.cli:cli',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.json'],
    },
    keywords='aruco markers computer-vision pose-estimation opencv tracking',
    project_urls={
        'Bug Reports': 'https://github.com/Jshulgach/ArUco_Pose_Estimation/issues',
        'Source': 'https://github.com/Jshulgach/ArUco_Pose_Estimation',
    },
)
