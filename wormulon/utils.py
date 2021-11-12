import io
import torch
from google.cloud import storage


class JobStatus:
    RUNNING = 0
    SUCCESS = 1
    FAILURE = 2
    ABORTED = 3

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


def _upload_data_to_gcs(bucket, remote_file_path, data):
    """Uploads a blob to GCS bucket"""
    client = storage.Client()
    blob = storage.blob.Blob.from_string(bucket + "/" + remote_file_path)
    blob.bucket._client = client
    blob.upload_from_string(data)


def _read_blob_gcs(bucket, remote_file_path):
    """Downloads a file from GCS to local directory"""
    client = storage.Client()
    bucket = client.get_bucket(bucket)
    blob = bucket.get_blob(remote_file_path)
    bytes = blob.download_as_bytes()
    buffer = io.BytesIO(bytes)
    return buffer
