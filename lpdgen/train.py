import argparse
from models import *
from io import *


if __name__ == "__main__":

    # Create a root parser
    root_parser = argparse.ArgumentParser(add_help=False)

    # Create subparsers for each model 
    subparsers = root_parser.add_subparsers(dest='model')

    # Models
    dcgan3d_parser = subparsers.add_parser('dcgan3d')

    # Parse the root args
    # This will pick up which command was run
    root_args, func_args  = root_parser.parse_known_args()

    # Run the training for a given model
    if root_args.model == 'dcgan3d':
        dcgan.cli_main(func_args, dcgan3d_parser)
        print("done")
    else:
        raise ValueError(f'Unknown model {root_args.model}')
