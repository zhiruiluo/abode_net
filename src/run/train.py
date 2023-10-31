from argparse import ArgumentParser, Namespace
import pytorch_lightning as pl

def get_parser() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('')
    
    return parser.parse_args()

def train(args):
    
    pl.Trainer()

def main():
    args = get_parser()
    train(args)