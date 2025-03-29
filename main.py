import argparse

from dotenv import load_dotenv


load_dotenv()


def main(args):
    pass


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--reload_data", action="store_true")
    return argparser.parse_args()


if __name__ == "__main__":
    main(parse_args())
