# -*- coding: utf-8 -*-
import sys

from setuptools import setup, find_packages

# Avoids IDE errors, but actual version is read from version.py
__version__ = ""
exec(open('similarities/version.py').read()) #exec() 函数则执行这个字符串作为 Python 代码。这意味着 version.py 文件中的所有代码都将被执行

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required.')

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='similarities',
    version=__version__,
    description='Similarities is a toolkit for compute similarity scores between two sets of strings.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/similarities',
    license="Apache License 2.0",
    zip_safe=False,
    python_requires=">=3.6.0",
    entry_points={"console_scripts": ["similarities = similarities.cli:main"]},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords='similarities,Chinese Text Similarity Calculation Tool,similarity,word2vec',
    install_requires=[
        "text2vec>=1.2.9",
        "jieba>=0.39",
        "loguru",
        "Pillow",
        "fire",
        "autofaiss",
        "transformers",
    ],
    packages=find_packages(),
)
