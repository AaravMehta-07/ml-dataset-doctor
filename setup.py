from setuptools import setup, find_packages

setup(
    name="ml-dataset-doctor",
    version="0.3.0",
    author="Aarav Mehta",
    description="Diagnose, fix, and audit ML datasets before training",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AaravMehta-07/ml-dataset-doctor",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn"
    ],
    entry_points={
        "console_scripts": [
            "dataset-doctor=dataset_doctor.cli:main"
        ]
    },
    python_requires=">=3.8",
)
