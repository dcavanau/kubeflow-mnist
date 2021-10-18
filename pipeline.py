
import kfp.dsl as dsl
from kfp.dsl import PipelineVolume
from kubernetes.client import V1VolumeMount
from kubernetes.client import V1Volume
from kubernetes.client import V1SecretVolumeSource

# To compile the pipeline:
#   dsl-compile --py pipeline.py --output pipeline.tar.gz
from constants import PROJECT_ROOT, CONDA_PYTHON_CMD


def git_clone_op(repo_url: str, pvolume: PipelineVolume):
    image = 'alpine/git:latest'

    commands = [
        f"git clone {repo_url} {PROJECT_ROOT}",
        f"cd {PROJECT_ROOT}"]

    op = dsl.ContainerOp(
        name='git clone',
        image=image,
        command=['sh'],
        arguments=['-c', ' && '.join(commands)],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume}
    )

    return op


def preprocess_op(image: str, pvolume: PipelineVolume, data_dir: str):
    return dsl.ContainerOp(
        name='preprocessing',
        image=image,
        command=[CONDA_PYTHON_CMD, f"{PROJECT_ROOT}/preprocessing.py"],
        arguments=["--data_dir", data_dir],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume}
    )


def train_and_eval_op(image: str, pvolume: PipelineVolume, data_dir: str):
    return dsl.ContainerOp(
        name='training and evaluation',
        image=image,
        command=[CONDA_PYTHON_CMD, f"{PROJECT_ROOT}/train.py"],
        arguments=["--data_dir", data_dir],
        file_outputs={'output': f'{PROJECT_ROOT}/output.txt'},
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={"/workspace": pvolume}
    )


def packaging(image: str, pvolume: PipelineVolume, model_path: str, model_name:str):
    return dsl.ContainerOp(
        name='packaging',
        image=image,
        command=[CONDA_PYTHON_CMD, f"{PROJECT_ROOT}/package.py"],
        arguments=["--model_path", model_path, "--model_name", model_name],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        file_outputs={'output': f'/workspace/kubeflow-mnist.tar'},
        pvolumes={"/workspace": pvolume}
    )


@dsl.pipeline(
    name='Fashion MNIST Training Pipeline',
    description='Fashion MNIST Training Pipeline to be executed on KubeFlow.'
)
def training_pipeline(image: str = 'dcavanau/kubeflow-mnist',
                      repo_url: str = 'https://61acc7bc6d89fb89dffb2c7e2142adffef6b13f1:x-oauth-basic@github.com/dcavanau/kubeflow-mnist.git',
                      data_dir: str = '/workspace'):

    _volume_op = dsl.VolumeOp(
        name="create pipeline volume",
        resource_name="pipeline-pvc",
        modes=dsl.VOLUME_MODE_RWM,
        size="3Gi",
        storage_class="default"
    )

    _git_clone = git_clone_op(repo_url=repo_url, pvolume=_volume_op.volume)

    _preprocess_data = preprocess_op(image=image,
                                    pvolume=_git_clone.pvolume,
                                    data_dir=data_dir).after(_git_clone)

    _training_and_eval = train_and_eval_op(image=image,
                                           pvolume=_preprocess_data.pvolume,
                                           data_dir=data_dir).after(_preprocess_data)

    _packaging = packaging(image=image,
                           pvolume=_training_and_eval.pvolume,
                           model_path=data_dir,
                           model_name='kubeflow-mnist').after(_training_and_eval)


if __name__ == '__main__':
    import kfp.compiler as compiler

    compiler.Compiler().compile(training_pipeline, __file__ + '.tar.gz')
