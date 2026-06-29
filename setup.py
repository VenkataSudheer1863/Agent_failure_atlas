from setuptools import setup, find_packages

setup(
    name="agent-failure-atlas",
    version="1.0.0",
    description="Open taxonomy, dataset, and benchmark for analyzing failure modes in autonomous AI agents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "requests>=2.31.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.1",
    ],
    extras_require={
        "experiments": [
            "openai>=1.30.0",
            "pandas>=2.1.0",
            "numpy>=1.26.0",
            "scipy>=1.11.0",
            "scikit-learn>=1.3.0",
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.26.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
