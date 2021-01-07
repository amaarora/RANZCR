import sys
import importlib
from types import SimpleNamespace
import argparse
import logging
logging.getLogger().setLevel(logging.INFO)

sys.path.append("../configs")

parser = argparse.ArgumentParser(description='')
parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)
logging.info(f"Using config file {parser_args.config}")

mod  = importlib.import_module(parser_args.config)
args = mod.args
args["experiment_name"] = parser_args.config

# allow for `args.{attribute}`
args =  SimpleNamespace(**args)