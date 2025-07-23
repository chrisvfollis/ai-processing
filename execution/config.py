# standard dependencies
import argparse

# 3rd-party dependencies
pass

# internal dependencies
pass


def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log-level', type=int, default=0)

    parser.add_argument('--retain-footage', action='store_true', default=False)
    parser.add_argument('--save-all-data', action='store_true', default=False)

    parser.add_argument('--start-from', type=str, help='Comma-separated datetime')
    parser.add_argument('--priority-cam', type=str)
    parser.add_argument('--f-cutoff', type=int, default=None)

    parser.add_argument('--id-strategy', type=str, default='assess_presence')

    return parser
