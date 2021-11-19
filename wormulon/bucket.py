import io
from collections import defaultdict
from google.cloud import storage
from wormulon.utils import JobState


class Bucket(object):
    def __init__(self, name):
        self.name = name

    def list(self, filter: str):
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(self.name)
        if filter:
            blobs = [blob for blob in blobs if filter in blob.name]
        return blobs

    def list_jobs(self, filter: JobState, limit: int = None, verbose: bool = True):
        results = defaultdict(list)
        jobs = self.list(filter="job")
        for job in jobs:
            results[job.state].append(job)

        if verbose:
            s = ""
            for k, v in results.items():
                s += f"{k}: {len(v)}, "
            print(s)

        return results[filter][:limit]

    def upload(self, path, data, overwrite=False):
        """Uploads a blob to GCS bucket"""
        if self.exists(path) and not overwrite:
            print(f"{path} already exists")
            return
        print(f"Uploading {self.name}/{path}")
        client = storage.Client()
        blob = storage.blob.Blob.from_string("gs://" + self.name + "/" + path)
        blob.bucket._client = client
        blob.upload_from_string(data)

    def download(self, path):
        """Downloads a file from GCS to local directory"""
        client = storage.Client()
        bucket = client.get_bucket(self.name)
        if path.startswith("gs://"):
            remote_file_path = path[5:]
        if remote_file_path.endswith("/"):
            remote_file_path = remote_file_path[:-1]
        if remote_file_path.startswith("/"):
            remote_file_path = remote_file_path[1:]
        blob = bucket.get_blob(remote_file_path)
        bytes = blob.download_as_bytes()
        buffer = io.BytesIO(bytes)
        return buffer

    def exists(self, path):
        """Downloads a file from GCS to local directory"""
        client = storage.Client()
        bucket = client.get_bucket(self.name)
        return bucket.blob(path).exists()

    def delete(self, path):
        client = storage.Client()
        bucket = client.get_bucket(self.name)
        return bucket.blob(path).delete()
