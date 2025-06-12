from setuptools import setup, find_packages

setup(
    name="harvestiq-ml-service",
    version="1.0.0",
    description="Machine Learning service for maize yield prediction",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.15.0",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
    ],
)
