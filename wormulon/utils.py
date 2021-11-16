import io
import subprocess
from google.cloud import storage
from typing import List


class JobStatus:
    RUNNING = 0
    SUCCESS = 1
    FAILURE = 2
    ABORTED = 3
    TIMEOUT = 4
    STARTING = 5
    PREEMPTED = 6

    @staticmethod
    def to_string(status):
        if status == JobStatus.RUNNING:
            return "RUNNING"
        elif status == JobStatus.SUCCESS:
            return "SUCCESS"
        elif status == JobStatus.FAILURE:
            return "FAILURE"
        elif status == JobStatus.ABORTED:
            return "ABORTED"
        else:
            return "UNKNOWN"


def _upload_data_to_gcs(bucket_name, remote_file_path, data):
    """Uploads a blob to GCS bucket"""
    client = storage.Client()
    blob = storage.blob.Blob.from_string("gs://" + bucket_name + "/" + remote_file_path)
    blob.bucket._client = client
    blob.upload_from_string(data)


def _read_blob_gcs(bucket, remote_file_path):
    """Downloads a file from GCS to local directory"""
    client = storage.Client()
    bucket = client.get_bucket(bucket)
    if remote_file_path.startswith("gs://"):
        remote_file_path = remote_file_path[5:]
    if remote_file_path.endswith("/"):
        remote_file_path = remote_file_path[:-1]
    if remote_file_path.startswith("/"):
        remote_file_path = remote_file_path[1:]
    blob = bucket.get_blob(remote_file_path)
    bytes = blob.download_as_bytes()
    buffer = io.BytesIO(bytes)
    return buffer


def _check_exists_gcs(bucket, remote_file_path):
    """Downloads a file from GCS to local directory"""
    client = storage.Client()
    bucket = client.get_bucket(bucket)
    return bucket.blob(remote_file_path).exists()


def _delete_blob_gcs(bucket, remote_file_path):
    """Downloads a file from GCS to local directory"""
    client = storage.Client()
    bucket = client.get_bucket(bucket)
    return bucket.blob(remote_file_path).delete()


def execute(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, error = process.communicate()
    return output, error


def get_tpu_ids(zone="us-central1-f") -> List[int]:
    command = f"gcloud alpha compute tpus list --zone={zone} --format=value[seperator=','](name)"
    output, error = execute(command.split())
    ids = output.decode("utf-8").split("\n")
    ids.remove("")
    int_ids = [-1]
    int_ids.extend([int(i.split("-")[-1]) for i in ids])
    return int_ids, error
