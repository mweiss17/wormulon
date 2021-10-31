import numpy as np
from ast import literal_eval

def uniform(bottom, top):
    return np.random.uniform(bottom, top)

def log_uniform(bottom, top):
    return np.exp(np.random.uniform(np.log(bottom), np.log(top)))

def discrete_uniform(choices):
    return np.random.choice(tuple(*choices))

def normal(mu, sigma):
    return np.random.normal(mu, sigma)

def log_normal(mu, sigma):
    return np.random.lognormal(mu, sigma)

def get_dist(dist_name, params):
    # takes a distribution name and a comma seperated list of string arguments and returns dist and python params
    mapping = {
        "discrete_uniform": discrete_uniform,
        "uniform": uniform,
        "log_uniform": log_uniform,
        "normal": normal,
        "log_normal": log_normal,
    }
    dist = mapping[dist_name]
    params = [literal_eval(p) for p in params.split(",")]
    return dist, params