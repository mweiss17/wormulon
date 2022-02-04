#!/usr/bin/env python

from copy import deepcopy
import sys
import os
from runpy import run_path
import rich
import uuid
import time
import click
import submitit
import pathlib
from submitit.core.utils import CommandFunction

try:
    from raven.core import RavenJob as Job
    from raven.utils import JobState
except ImportError:
    from wormulon.core import Job
    from wormulon.utils import JobState


class Salvo(object):
    def __init__(self):
        self.executor = None
        # Args
        self._argv = None
        self._salvo_arg_block = None
        self._script_arg_block = None
        self._param_generator_arg_block = None
        # Other variables
        self._salvo_args = None
        self._script_path = None
        self._script_args = None
        self._param_generator_path = None
        self._param_generator_args = None

    def _build_executor(self, experiment_directory):
        # create the submitit executor for creating and managing jobs
        executor = submitit.AutoExecutor(
            folder=os.path.join(experiment_directory, "Logs")
        )

        # setup the executor parameters based on the cluster location
        if executor.cluster == "slurm":
            executor.update_parameters(
                mem_gb=self.get_salvo_arg("mem_gb", 16),
                cpus_per_task=self.get_salvo_arg("cpus_per_task", 12),
                timeout_min=self.get_salvo_arg("timeout_min", 1000),
                tasks_per_node=1,
                nodes=1,
                slurm_partition=self.get_salvo_arg("slurm_partition", "long"),
                gres=self.get_salvo_arg("gres", "gpu:rtx8000:1"),
            )
        return executor

    @property
    def argv(self):
        if self._argv is None:
            self.record_args()
        return self._argv

    def set_argv(self, args):
        self._argv = list(args)
        return self

    @property
    def argv_seperator_idxs(self):
        seperator_idxs = [idx for idx, arg in enumerate(self.argv) if arg == "---"]
        if len(seperator_idxs) == 0:
            raise ValueError("Could not find a --- seperator.")
        elif len(seperator_idxs) == 1:
            # This means the command was launched as:
            #   python salvo.py script_path arg1 arg2 --- generator.py
            # We pretend there was another sep at the index = 0, equivalent to:
            #   python salvo.py --- script_path arg1 arg2 --- generator.py
            seperator_idxs = [0] + seperator_idxs
        elif len(seperator_idxs) > 2:
            raise ValueError("Was expecting a max of two --- seperators.")
        # If we make it here, the command was launched as:
        #   python salvo.py salvo_arg --- script_path arg1 arg2 --- generator.py
        return seperator_idxs

    def record_args(self):
        self._argv = sys.argv
        return self

    def parse_arg_blocks(self, record=True):
        salvo_sep_idx, script_sep_idx = self.argv_seperator_idxs
        salvo_arg_block = self.argv[0:salvo_sep_idx]
        script_arg_block = self.argv[salvo_sep_idx + 1 : script_sep_idx]
        param_generator_arg_block = self.argv[script_sep_idx + 1 :]
        if record:
            self._salvo_arg_block = salvo_arg_block
            self._script_arg_block = script_arg_block
            self._param_generator_arg_block = param_generator_arg_block
        return salvo_arg_block, script_arg_block, param_generator_arg_block

    @property
    def salvo_arg_block(self):
        if self._salvo_arg_block is None:
            self.parse_arg_blocks(record=True)
        return self._salvo_arg_block

    @property
    def script_arg_block(self):
        if self._script_arg_block is None:
            self.parse_arg_blocks(record=True)
        return self._script_arg_block

    @property
    def param_generator_arg_block(self):
        if self._param_generator_arg_block is None:
            self.parse_arg_blocks(record=True)
        return self._param_generator_arg_block

    @property
    def script_path(self):
        if self._script_path is None:
            return self.script_arg_block[0]
        else:
            return self._script_path

    def set_script_path(self, script_path):
        self._script_path = script_path

    def parse_script_args(self, record=True):
        script_args = self.script_arg_block[1:]
        if record:
            self._script_args = script_args
        return script_args

    def parse_param_generator_path(self, record=True):
        param_generator_path = self.param_generator_arg_block[0]
        if record:
            self._param_generator_path = param_generator_path
        return param_generator_path

    def parse_param_generator_args(self, record=True):
        param_generator_args = self.param_generator_arg_block[1:]
        if record:
            self._param_generator_args = param_generator_args
        return param_generator_args

    @property
    def param_generator_args(self):
        if self._param_generator_args is None:
            self.parse_param_generator_args(record=True)
        return self._param_generator_args

    @property
    def script_args(self):
        if self._script_args is None:
            return self.parse_script_args(record=True)
        else:
            return self._script_args

    @property
    def param_generator_path(self):
        if self._param_generator_path is None:
            return self.parse_param_generator_path(record=True)
        else:
            return self._param_generator_path

    def in_salvo_arg_block(self, flag):
        if len(self.salvo_arg_block) > 0 and flag in self.salvo_arg_block:
            return True
        else:
            return False

    def get_salvo_arg(self, flag, default=None):
        if not self.in_salvo_arg_block(flag):
            return default
        arg_idx = self.salvo_arg_block.index(flag) + 1
        return self.salvo_arg_block[arg_idx]

    @property
    def is_dry_run(self):
        return self.in_salvo_arg_block("--dry")

    @property
    def use_abs_path(self):
        return self.in_salvo_arg_block("--use-abs-path")

    def get_generator(self):
        return run_path(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(), self.param_generator_path
            )
        )["generator"](self.param_generator_args)

    def launch_one_job(
        self,
        job_idx,
        kwargs,
        script_path,
        template_args,
        is_dry_run=False,
        use_abs_path=False,
    ):

        job_kwargs = {"job_idx": job_idx, "uuid": uuid.uuid4().hex, **kwargs}
        script_args = [arg.format(**job_kwargs) for arg in template_args]
        if use_abs_path:
            script_path = os.path.abspath(script_path)
        else:
            script_path = script_path

        if self.in_salvo_arg_block("--use-xvfb"):
            command = ['xvfb-run -a -s "-screen 0 800x600x24"', 'python3']
        else:
            command = ["python3"]

        rich.print(
            f":rocket: [bold]Launching:[/bold] [bold blue]{' '.join(command)}[/bold blue] "
            f"[bold green]{script_path}[/bold green] "
            f"[bold magenta]{' '.join(script_args)}[/bold magenta]"
        )

        if is_dry_run:
            job = None
        else:
            command.extend([script_path, *script_args])
            # Setup Submitit
            if self.executor is None:
                self.executor = self._build_executor(script_args[0])
            job = self.executor.submit(CommandFunction(command))

        job_info = {
            "job": job,
            "script_path": script_path,
            "script_args": script_args,
        }
        return job_info

    def launch(self, generator=None):
        # Launch the salvo!
        # Import the generator from the file
        if generator is None:
            generator = self.get_generator()
        template_args = deepcopy(self.script_args)

        launched_jobs = []
        for job_idx, kwargs in enumerate(generator):
            job_info = self.launch_one_job(
                job_idx,
                kwargs,
                self.script_path,
                template_args,
                is_dry_run=self.is_dry_run,
                use_abs_path=self.use_abs_path,
            )
            launched_jobs.append(job_info)
        # Store launched jobs if required
        log_path = self.get_salvo_arg("--log")
        if log_path is not None:
            rich.print(
                f":scroll: [bold]Dumping a list of launched jobs at:[/bold] "
                f"[bold green]{log_path}[/bold green]"
            )
            dump_yaml(launched_jobs, log_path)  # noqa

    def nuke(self, generator=None):
        if generator is None:
            generator = self.get_generator()
        for job_spec in generator:
            rich.print(
                f":boom: [bold]Nuking:[/bold] "
                f"[bold blue]{job_spec['job_id']} / "
                f"{job_spec.get('wandb_name', 'UNK')} / "
                f"{job_spec.get('wandb_id', 'UNK')}[/bold blue]"
            )
            if not self.is_dry_run:
                Job(job_spec["job_id"]).nuke()

    def request_exit(self, generator=None):
        if generator is None:
            generator = self.get_generator()
        for job_spec in generator:
            rich.print(
                f":pleading_face: [bold]Requesting exit:[/bold] "
                f"[bold blue]{job_spec['job_id']} / "
                f"{job_spec.get('wandb_name', 'UNK')} / "
                f"{job_spec.get('wandb_id', 'UNK')}[/bold blue]"
            )
            if not self.is_dry_run:
                Job(job_spec["job_id"]).request_exit()
                submit_job()

    def submit_job(self, executor, script_path, script_args):
        pass

    def nuke_and_launch(self, generator=None):
        if generator is None:
            generator = self.get_generator()
        generator = list(generator)
        # First, nuke
        self.nuke(generator)
        rich.print("[red]-------------------[/red]")
        if self.in_salvo_arg_block("--wait-before-launch"):
            console = rich.get_console()
            with console.status("Waiting [bold]60s[/bold] for launch..."):
                time.sleep(60)
        # Next, launch
        self.launch(generator)

    def fire(self):
        if self.in_salvo_arg_block("--nuke"):
            self.nuke()
        elif self.in_salvo_arg_block("--request-exit"):
            self.request_exit()
        elif self.in_salvo_arg_block("--nuke-and-launch"):
            self.nuke_and_launch()
        else:
            self.launch()


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
def fire():
    Salvo().fire()


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
def nuke():
    Salvo().nuke()


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
def snipe():
    Salvo().snipe()
