from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

optional_packages = {
    "tf" : ['tensorflow>=2.2.0', 'tensorflow-text', 'tensorflow-hub'],
    "manticore": ['manticoresearch==1.0.6', 'pandas', 'typer', 'wasabi']
}

setup(
    name="beir",
    version="1.0.1",
    author="Nandan Thakur",
    author_email="nandant@gmail.com",
    description="A Heterogeneous Benchmark for Information Retrieval",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url="https://github.com/beir-cellar/beir",
    download_url="https://github.com/beir-cellar/beir/archive/v1.0.1.zip",
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'sentence-transformers',
        'pytrec_eval',
        'faiss_cpu',
        'elasticsearch==7.9.1',
        'datasets'
    ],
    extras_require = optional_packages,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="Information Retrieval Transformer Networks BERT PyTorch IR NLP deep learning"
)