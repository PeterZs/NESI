import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging

from trainers.nesi_trainer import NESITrainer
from options import parse_options


def configure_logging(args):
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt='[%(asctime)s] [INFO] %(message)s', datefmt='%d/%m %H:%M:%S')
    logger_handler.setFormatter(formatter)
    logger.addHandler(logger_handler)


if __name__ == "__main__":
    """Main program."""

    args, args_str = parse_options()
    configure_logging(args)
    logging.info(f'Parameters: \n{args_str}')
    logging.info(f'Training on {args.dataset_path}')
    model = NESITrainer(args, args_str)
    model.train()
