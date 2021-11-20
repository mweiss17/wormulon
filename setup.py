#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="wormulon",
    description="Wormulon: Home planet of the Slurm Worms",
    version="0.1.0",
    author="Martin Weiss; Nasim Rahaman",
    author_email="martin.clyde.weiss@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["click", "rich", "submitit", "wandb", "numpy", "stopit"],
    entry_points={
        "console_scripts": [
            "submit = wormulon.submit:main",
            "salvo = wormulon.salvo:fire",
            "tpu_train = wormulon.tpu_runner:__init__",
        ],
    },
)
