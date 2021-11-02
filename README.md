# [Wormulon](https://futurama.fandom.com/wiki/Slurm)
This package exists to support machine learning projects that run on clusters. It makes interacting with [Slurm](https://slurm.schedmd.com/documentation.html) less of a nightmare through the cunning use of [submitit](https://github.com/facebookincubator/submitit) and [wandb](https://wandb.ai/).

## Installation
`git clone git@github.com:mweiss17/wormulon.git`
`cd wormulon`
`pip install -e .`

## Usage
Use Wormulon after you've developed your models and made a vague pass at debugging your scripts -- once you are ready to start burning up the cluster. Instead of writing a crappy bash script, or dropping in some home-rolled project-specific scripts, just run:
`submit scripts/<your-train-script>.py experiments/<exp-1> templates/<template-1>`

If you want to do a hyper-parameter search, then run:

`salvo -dry --- scripts/<your-train-script>.py ~/scratch/experiments/exp1-{job_idx} --inherit templates/exp1 --config.use_wandb True --config._wandb_group group-1 --config._wandb_run_name exp1-{job_idx} --config.seed {job_idx} --- generators/random_search.py -d lr~log_uniform[0.01,0.001] -n 10`


![slurm](https://user-images.githubusercontent.com/2440148/139599621-a049cf4a-2ddd-4694-b598-af83c47fe15c.png)
