import click
from pathlib import Path
from wormulon.tpu.tpu import TPU
from wormulon.tpu.bucket import Bucket
from wormulon.utils import JobState, execute, dump_yaml


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
@click.option("--wipe")
def cleanup_jobs(bucket_name, filter=None, wipe=False):
    bucket = Bucket(bucket_name)
    if filter:
        filter = JobState[filter]
    jobs = bucket.list_jobs(filter)
    for job in jobs:
        job_id = Path(job['blob'].name).parent
        if wipe:
            print("wiping the directory.")
            bucket.delete_folder(bucket_name, job_id)
        else:
            print(f"setting {job_id} to FAILURE.")
            bucket.upload(job['blob'].name, dump_yaml({"state": JobState.FAILURE.value}), overwrite=True)


@click.command(context_settings={})
@click.argument("bucket_name")
def show_experiments(bucket_name):
    bucket = Bucket(bucket_name)
    bucket.list_experiments()


@click.command(context_settings={})
def show_tpus():
    zones = ["us-central1-f", "europe-west4-a"]
    for zone in zones:
        command = f"gcloud compute tpus list --format=value(NAME,STATUS) --zone {zone}"
        stdout, stderr, retcode = execute(command.split(), capture_output=True)
        rows = stdout.split("\n")
        rows.remove("")
        for row in rows:
            print(row + f"\t {zone}")


@click.command(context_settings={})
def delete_all_tpus():
    zones = ["us-central1-f", "europe-west4-a"]
    default_tpu_kwargs = {
            "network": "tpu-network",
            "subnet": "swarm-2",
            "netrange": "192.170.0.0/29",
            "acc_type": "v3-8",
            "preemptible": False,
            "bucket": "must-results",
            "project": "polytax"
        }

    for zone in zones:
        command = f"gcloud compute tpus list --format=value(NAME,STATUS) --zone {zone}"
        stdout, stderr, retcode = execute(command.split(), capture_output=True)
        rows = stdout.split("\n")
        rows.remove("")
        default_tpu_kwargs["zone"] = zone
        for row in rows:
            if row.split("\t")[1] == "READY":
                tpu = TPU(row.split("\t")[0], **default_tpu_kwargs)
                tpu.delete()

