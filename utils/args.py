import argparse


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--reload_data", action="store_true")
    return argparser.parse_args()
