from argparse import ArgumentParser
import wandb
import os

try:
    from raven.core import RavenJob as Job
    from raven.utils import JobStatus
except ImportError:
    from wormulon.core import Job
    from wormulon.utils import JobStatus

def parse_args(args: list):
    parsey = ArgumentParser()
    parsey.add_argument("-s", "--sweep-name", required=True, type=str)
    parsey.add_argument("-e", "--wandb-entity", required=True, type=str)
    parsey.add_argument("-p", "--wandb-project", required=True, type=str)
    parsey.add_argument("-ma", "--max-age", default=169200, type=int)
    parsey.add_argument(
        "-rjo", "--running-jobs-only", action="store_true", default=False
    )
    parsey.add_argument(
        "-nrjo", "--not-running-jobs-only", action="store_true", default=False
    )
    parsey.add_argument("-ms", "--min-steps", default=None, type=int)
    parsey.add_argument("-mxs", "--max-steps", default=None, type=int)
    parsey.add_argument(
        "-f", "--filter-name", default="speedrun_meta__sweep_name", type=str
    )
    parsey.add_argument("-mna", "--min-age", default=float("inf"), type=int)
    return parsey.parse_args(args)

def generator(args: list):
    args = parse_args(args)
    api = wandb.Api()
    runs = api.runs(
        path=f"{args.wandb_entity}/{args.wandb_project}",
        filters={"$or": [{f"config.{args.filter_name}": args.sweep_name}]},
    )
    for run in runs:
        if (
            args.min_steps is not None
            and run.summary.get("step") is not None
            and run.summary.get("step") < args.min_steps
        ):
            # This run is not far enough in to training, so we discard it as requested.
            continue
        if (
            args.max_steps is not None
            and run.summary.get("step") is not None
            and run.summary.get("step") > args.max_steps
        ):
            # This run is too far in to training, so we discard it as requested
            continue
        # Get specs from raven
        try:
            job_directory = run.config["speedrun_meta__job_directory"]
        except KeyError:
            continue
        job_id = os.path.basename(job_directory)
        job = Job(job_id)
        last_heartbeat_at = job.last_heartbeat_at(relative_to_now=True)
        if args.running_jobs_only or args.not_running_jobs_only:
            if args.running_jobs_only:
                # Criterion satisfied only if job is running
                is_running_criterion = job.get_status() == JobStatus.RUNNING
            else:
                is_running_criterion = job.get_status() != JobStatus.RUNNING
        else:
            # Criterion satisfied irrespective of whether job is running or not
            is_running_criterion = True
        if (
            last_heartbeat_at is not None
            and args.max_age < last_heartbeat_at < args.min_age
            and is_running_criterion
        ):
            yield {"job_id": job_id, "wandb_id": run.id, "wandb_name": run.name}
