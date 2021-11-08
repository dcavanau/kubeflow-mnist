import calendar
import os
import subprocess
import sys

import argparse

from constants import PROJECT_NAME
from constants import PROJECT_BASE
from constants import MODEL_VERSION


def package(model_path: str, model_name:str, model_version: str):

    # Set up the paths
    if model_path is None:
        model_path = PROJECT_BASE

    if model_name is None:
        model_name = PROJECT_NAME

    if model_version is None:
        model_version = MODEL_VERSION

    package_name = os.path.join('/', model_path, model_name)
    package_name = package_name + '.tar.gz'
    tar_command = 'tar -czvf  %s -C %s %s/%s' % (package_name, model_path, model_name, str(int(model_version)))
    subprocess.run(tar_command, shell=True, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kubeflow MNIST packaging script')
    parser.add_argument('--model_path', help='base folder to export model')
    parser.add_argument('--model_name', help='model name to append to the model path')
    parser.add_argument('--model_version', help='model version to append to the model path')
   args = parser.parse_args()

    package(model_path=args.model_path, model_name=args.model_name, model_version=args.model_version)
