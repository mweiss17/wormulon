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
    install_requires=["click", "rich", "submitit", "wandb", "numpy", "stopit", "google-cloud-storage", "dill", "gcloud"],
    entry_points={
        "console_scripts": [
            "submit = wormulon.submit:main",
            "tpu_submit = wormulon.tpu.submit:main",
            "tpu_nanny = wormulon.tpu.nanny:main",
            "salvo = wormulon.salvo:fire",
            "tpu_salvo = wormulon.tpu.salvo:fire",
            "tpu_train = wormulon.tpu.tpu_runner:main",
            "show_jobs = wormulon.tpu.utils:show_jobs",
            "delete_jobs = wormulon.tpu.utils:delete_jobs",
            "show_experiments = wormulon.tpu.utils:show_experiments",
            "show_tpus = wormulon.tpu.utils:show_tpus",
            "delete_all_tpus = wormulon.tpu.utils:delete_all_tpus",
            "nuke_exps = wormulon.tpu.utils:nuke_exps"
        ],
    },
)
