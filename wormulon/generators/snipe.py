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
    parsey.add_argument(
        "-f", "--filter-name", default="speedrun_meta__sweep_name", type=str
    )
    parsey.add_argument(
        "-rjo", "--running-jobs-only", action="store_true", default=False
    )
    parsey.add_argument(
        "-nrjo", "--not-running-jobs-only", action="store_true", default=False
    )
    parsey.add_argument("-ms", "--min-steps", default=None, type=int)
    parsey.add_argument("-mxs", "--max-steps", default=None, type=int)
    parsey.add_argument("-ma", "--min-age", default=None, type=int)
    parsey.add_argument("-mxa", "--max-age", default=None, type=int)
    return parsey.parse_args(args)


def main(args: list):
    args = parse_args(args)
    api = wandb.Api()
    runs = api.runs(
        path=f"{args.wandb_entity}/{args.wandb_project}",
        filters={"$or": [{f"config.{args.filter_name}": args.sweep_name}]},
    )
    for run in runs:
        step = run.summary.get("step")
        if step is not None:
            if args.min_steps is not None and args.max_steps is not None:
                # We're keeping this run only if step is in the right range
                if args.min_steps < step < args.max_steps:
                    pass
                else:
                    continue
            elif args.min_steps is not None:
                # We're keeping this run only if step is above min_steps
                if step > args.min_steps:
                    pass
                else:
                    continue
            elif args.max_steps is not None:
                if step < args.max_steps:
                    pass
                else:
                    continue
            else:
                # both min-steps and max-steps are not provided, so we assume we
                # can proceed
                pass
        else:
            # step is None, so there's nothing we can do for these runs
            continue

        # Get specs from raven
        try:
            job_directory = run.config["speedrun_meta__job_directory"]
        except KeyError:
            # Again, nothing we can do for these runs
            continue
        # Figure out age criterion
        job_id = os.path.basename(job_directory)
        job = Job(job_id)
        last_heartbeat_at = job.last_heartbeat_at(relative_to_now=True)
        if last_heartbeat_at is not None:
            if args.max_age is not None and args.min_age is not None:
                # Keep only for age in the right range
                if args.min_age < last_heartbeat_at < args.max_age:
                    age_criterion = True
                else:
                    age_criterion = False
            elif args.min_age is not None:
                if last_heartbeat_at > args.min_age:
                    age_criterion = True
                else:
                    age_criterion = False
            elif args.max_age is not None:
                if last_heartbeat_at < args.max_age:
                    age_criterion = True
                else:
                    age_criterion = False
            else:
                # both min and max age is None, so we assume age_criterion is True
                age_criterion = True
        else:
            # No heartbeat recorded, so we continue
            age_criterion = True
        # Figure out is_running_criterion
        if args.running_jobs_only or args.not_running_jobs_only:
            if args.running_jobs_only:
                # Criterion satisfied only if job is running
                is_running_criterion = job.get_status() == JobStatus.RUNNING
            else:
                is_running_criterion = job.get_status() != JobStatus.RUNNING
        else:
            # Criterion satisfied irrespective of whether job is running or not
            is_running_criterion = True

        if age_criterion and is_running_criterion:
            yield {"job_id": job_id, "wandb_id": run.id, "wandb_name": run.name}
