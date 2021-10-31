import numpy as np
from argparse import ArgumentParser
from wormulon import dists


def generator(args):
    parser = ArgumentParser()
    parser.add_argument("-d", "--distributions", type=str, required=True)
    parser.add_argument("-n", "--num-runs", type=int, required=True)
    parsed = parser.parse_args(args)

    dists_to_sample = {}
    for dist_string in parsed.distributions.split("+"):
        hyper_name, dist_info = dist_string.split("~")
        dist_name, params = dist_info.split("[")
        dists_to_sample[hyper_name] = dists.get_dist(dist_name, params[:-1])

    for _ in range(parsed.num_runs):
        sampled = {dist_name: dist_fn(*params) for dist_name, (dist_fn, params) in dists_to_sample.items()}
        yield sampled
