import argparse
from pathlib import Path


def mod_arg_parser(desc: str, package: str, file_name: str) -> str:
    parser = argparse.ArgumentParser(description=desc, usage='python -m {}.{} [-h] config_folder_path'
                                     .format(package, Path(file_name).stem))
    parser.add_argument('config_path', metavar='config_folder_path', type=str, help='Path to configuration folder')
    args = parser.parse_args()
    return args.config_path
