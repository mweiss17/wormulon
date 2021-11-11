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


def _upload_blob_to_gcs_from_path(local_path, gcs_path):
    """Uploads a file to GCS bucket"""
    client = storage.Client()
    blob = storage.blob.Blob.from_string(gcs_path)
    blob.bucket._client = client
    blob.upload_from_filename(local_path)


def _upload_data_to_gcs(data, gcs_path):
    """Uploads a blob to GCS bucket"""
    client = storage.Client()
    blob = storage.blob.Blob.from_string(gcs_path)
    blob.bucket._client = client
    blob.upload_from_string(data)


def _read_blob_gcs(bucket, remote_file_path, destination):
    """Downloads a file from GCS to local directory"""
    client = storage.Client()
    bucket = client.get_bucket(bucket)
    blob = bucket.get_blob(remote_file_path)
    blob.download_to_filename(destination)
    return os.read(destination)
