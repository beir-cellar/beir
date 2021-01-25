from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="beir",
    version="0.0.2",
    author="Nandan Thakur",
    author_email="nandant@gmail.com",
    description="A Heterogeneous Benchmark for Information Retrieval",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/beir-nlp/beir",
    download_url="https://github.com/beir-nlp/beir/archive/v0.0.1.zip",
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'sentence-transformers',
        'pytrec_eval'
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="Information Retrieval Transformer Networks BERT PyTorch IR NLP deep learning"
)