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
