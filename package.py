import calendar
import os
import subprocess

import argparse

from constants import PROJECT_NAME
from constants import PROJECT_BASE


def package(model_path: str, model_name:str):

    # Set up the paths
    if model_path is None:
        model_path = PROJECT_BASE

    if model_name is None:
        model_name = PROJECT_NAME

    package_name = os.path.join('/', model_path, model_name)
    package_name = package_name + '.tar'

    subprocess.run(f'ar -cjf {package_name} -C {model_path} {model_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kubeflow FMNIST packaging script')
    parser.add_argument('--model_path', help='base folder to export model')
    parser.add_argument('--model_name', help='model name to append to the model path')
    args = parser.parse_args()

    package(model_path=args.model_path, model_name=args.model_name)
