from setuptools import setup, find_packages

setup(
    name="fh_prevail",
    version="0.1.1",
    author="Sina Mirshahi",
    author_email="sina7th@gmail.com",
    description="A time-series forecasting model using UR2CUTE.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/FH-Prevail/UR2CUTE",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow",
        "scikit-learn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
