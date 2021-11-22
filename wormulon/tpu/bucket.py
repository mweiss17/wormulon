import io
from collections import defaultdict
from google.cloud import storage
from wormulon.utils import JobState, load_yaml


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
        blobs = self.list(filter="jobstate")
        for blob in blobs:
            bytes = blob.download_as_bytes()
            buffer = io.BytesIO(bytes)
            jobstate = load_yaml(buffer.getvalue())
            results[JobState(jobstate.get("state")).name].append(jobstate)

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
            path = path[5:]
        if path.endswith("/"):
            path = path[:-1]
        if path.startswith("/"):
            path = path[1:]
        blob = bucket.get_blob(path)
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

    def delete_all(self, path):
        client = storage.Client()
        blobs = client.list_blobs(self.name, prefix=path)
        for blob in blobs:
            blob.delete()

    def touch(self, path):
        self.upload(path, "")
