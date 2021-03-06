import io
import operator
import time
from datetime import datetime
from dateutil import parser
from collections import defaultdict, namedtuple
from google.cloud import storage
from wormulon.utils import JobState, load_yaml
from wormulon.tpu.fncall import FunctionCall
from wormulon.train_state import TrainState
Experiment = namedtuple("Experiment", ["experiment_name", "dataset_name", "step", "blob"])

class Bucket(object):
    def __init__(self, name):
        self.name = name
        self.last_touch = time.time()

    def list(self, filter: str):
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(self.name)
        if filter:
            blobs = [blob for blob in blobs if filter in blob.name]
        return blobs

    def list_jobs(self, filters = []):
        results = []
        blobs = self.list(filter="jobstate")
        for blob in blobs:
            bytes = blob.download_as_bytes()
            buffer = io.BytesIO(bytes)
            jobstate = load_yaml(buffer.getvalue())
            jobstate['state'] = JobState(jobstate.get("state")).name
            jobstate['blob'] = blob

            if not filters:
                results.append(jobstate)
            for filter in filters:
                if jobstate.get("state") == filter.name:
                    results.append(jobstate)
        return results

    def list_experiments(self):

        blobs = self.list("trainstate")
        experiments = defaultdict(list)
        for blob in blobs:
            step_num = int(blob.name.split("-")[-1].split(".")[0])
            dataset_name = blob.name.split("/")[-1].split("-")[0]
            exp_name = blob.name.split("/")[1]
            experiments[f"{exp_name}-{dataset_name}"].append(Experiment(exp_name, dataset_name, step_num, blob))

        last_checkpoints = {}
        for exp_id in experiments.keys():
            exp = experiments[exp_id]
            exp.sort(key=operator.attrgetter("step"))
            last_checkpoints[exp_id] = exp[-1]

        print("Found the following experiments:\n")
        for exp_id, exp in last_checkpoints.items():
            updated = parser.parse(exp.blob._properties.get('updated', ''))
            print(f"{exp_id}: {exp.blob.name}, updated on {updated}")
        return last_checkpoints

    def get_latest_trainstate(self, experiment_directory):
        blobs = self.list(filter=f"{experiment_directory}/trainstate")
        blobs.sort(key=operator.attrgetter("updated"))
        bytes = blobs[-1].download_as_bytes()
        buffer = io.BytesIO(bytes)
        trainstate = TrainState.deserialize(buffer)
        return trainstate

    def get_latest_fncall(self, experiment_directory):
        blobs = self.list(filter=f"{experiment_directory}")
        fncalls = list(filter(lambda x: x.name.endswith("function_call.pkl"), blobs))
        fncalls.sort(key=operator.attrgetter("updated"))
        bytes = fncalls[-1].download_as_bytes()
        buffer = io.BytesIO(bytes)
        fncall = FunctionCall.deserialize(buffer)
        return fncall

    def delete_folder(self, bucket_name, folder):
        """
        This function deletes from GCP Storage

        :param bucket_name: The bucket name in which the file is to be placed
        :param folder: Folder name to be deleted
        :return: returns nothing
        """
        cloud_storage_client = storage.Client()
        bucket = cloud_storage_client.bucket(bucket_name)
        try:
            bucket.delete_blobs(blobs=list(bucket.list_blobs(prefix=folder)))
        except Exception as e:
            print(str(e.message))

    def upload(self, path, data, overwrite=False):
        """Uploads a blob to GCS bucket"""
        if self.exists(path) and not overwrite:
            print(f"{path} already exists")
            return
        print(f"Uploading to {self.name}/{path}")

        client = storage.Client()
        blob = storage.blob.Blob.from_string("gs://" + self.name + "/" + path)
        blob.bucket._client = client
        blob.upload_from_string(data)

    def download(self, path):
        blob = self.get_blob(path)
        bytes = blob.download_as_bytes()
        buffer = io.BytesIO(bytes)
        return buffer

    def get_blob(self, path):
        """gets a blob from GCS"""
        client = storage.Client()
        bucket = client.get_bucket(self.name)
        if path.startswith("gs://"):
            path = path[5:]
        if path.endswith("/"):
            path = path[:-1]
        if path.startswith("/"):
            path = path[1:]
        blob = bucket.get_blob(path)
        return blob

    def exists(self, path):
        """Downloads a file from GCS to local directory"""
        client = storage.Client()
        bucket = client.get_bucket(self.name)
        return bucket.blob(path).exists()

    def delete(self, path):
        client = storage.Client()
        bucket = client.get_bucket(self.name)
        try:
            bucket.blob(path).delete()
        except Exception:
            pass

    def delete_all(self, path):
        client = storage.Client()
        blobs = client.list_blobs(self.name, prefix=path)
        for blob in blobs:
            blob.delete()

    def touch(self, path):
        if time.time() - self.last_touch > 5:
            self.last_touch = time.time()
            self.upload(path, "", overwrite=True)
