from pathlib import Path
from argparse import ArgumentParser


def generator(args):
    parser = ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, required=True)
    parsed = parser.parse_args(args)

    for dir_path in Path(parsed.dir).iterdir():
        yield dict(dir_path=str(dir_path))
