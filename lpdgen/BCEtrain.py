import argparse
from models import BCE_dcgan
from io import *

if __name__ == "__main__":

    print("=========================")
    print(">>>>>>>>>>> BCE gan start")
    print("=========================")
    print()

    # Create a root parser
    root_parser = argparse.ArgumentParser(add_help=False)

    # Create subparsers for each model 
    subparsers = root_parser.add_subparsers(dest='model')

    # Models
    dcgan3d_parser = subparsers.add_parser('dcgan3d')

    # Parse the root args
    # This will pick up which command was run
    root_args, func_args  = root_parser.parse_known_args()

    # Run the training
    BCE_dcgan.cli_main(func_args, dcgan3d_parser)
    print("done")
