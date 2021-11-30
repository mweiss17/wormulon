import click
from pathlib import Path
from wormulon.tpu.bucket import Bucket
from wormulon.utils import JobState


@click.command(context_settings={})
@click.argument("bucket_name")
@click.option("--filter")
def show_jobs(bucket_name, filter=None):
    bucket = Bucket(bucket_name)
    if filter:
        filter = JobState[filter]
    bucket.list_jobs(filter)

@click.command(context_settings={})
@click.argument("bucket_name")
@click.option("--filter")
def cleanup_jobs(bucket_name, filter=None):
    bucket = Bucket(bucket_name)
    if filter:
        filter = JobState[filter]
    jobs = bucket.list_jobs(filter)
    for job in jobs:
        job_id = Path(job['blob'].name).parent
        print(f"deleting {job_id}")
        bucket.delete_folder(bucket_name, job_id)



@click.command(context_settings={})
@click.argument("bucket_name")
def show_experiments(bucket_name):
    bucket = Bucket(bucket_name)
    bucket.list_experiments()

