
import kfp.dsl as dsl
from kfp.dsl import PipelineVolume

# To compile the pipeline:
#   dsl-compile --py pipeline.py --output pipeline.tar.gz
from constants import PROJECT_ROOT, PROJECT_NAME, PROJECT_BASE, CONDA_PYTHON_CMD


def git_clone_op(repo_url: str):
    image = 'alpine/git:latest'

    commands = [
        f"git clone {repo_url} {PROJECT_ROOT} || git -C {PROJECT_ROOT} pull",
        f"cd {PROJECT_ROOT}"]

    volume_op = dsl.VolumeOp(
        name="create pipeline volume",
        resource_name="pipeline-pvc",
        modes=["ReadWriteOnce"],
        size="3Gi"
    )

    op = dsl.ContainerOp(
        name='git clone',
        image=image,
        command=['sh'],
        arguments=['-c', ' && '.join(commands)],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={f"/{PROJECT_BASE}": volume_op.volume}
    )

    return op


def preprocess_op(image: str, pvolume: PipelineVolume, data_dir: str):
    return dsl.ContainerOp(
        name='preprocessing',
        image=image,
        command=[CONDA_PYTHON_CMD, f"{PROJECT_ROOT}/preprocessing.py"],
        arguments=["--data_dir", os.path.join('/', PROJECT_BASE, data_dir)],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={f"/{PROJECT_BASE}": pvolume}
    )


def train_and_eval_op(image: str, pvolume: PipelineVolume, data_dir: str, model_path: str, model_name: str, model_version: int):
    return dsl.ContainerOp(
        name='training and evaluation',
        image=image,
        command=[CONDA_PYTHON_CMD, f"{PROJECT_ROOT}/train2.py"],
        arguments=["--data_dir", data_diros.path.join('/', PROJECT_BASE, data_dir), 
                   "--model_path", model_path, 
                   "--model_name", model_name, 
                   "--model_version", model_version],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={f"/{PROJECT_BASE}": pvolume}
    )


def packaging(image: str, pvolume: PipelineVolume, model_path: str, model_name: str, model_version: int):
    return dsl.ContainerOp(
        name='packaging',
        image=image,
        command=[CONDA_PYTHON_CMD, f"{PROJECT_ROOT}/package.py"],
        arguments=["--model_path", model_path, "--model_name", model_name, "--model_version", model_version],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={f"/{PROJECT_BASE}": pvolume}
    )

def publish(pvolume: PipelineVolume):
    image = 'dcavanau/kubeflow-horizon'
    commands = [
        f"cd /tmp",
        f"make publish-model",
        f"make publish-model-config"]
   
    return dsl.ContainerOp(
        name='publish',
        image=image,
        command=['sh'],
        arguments=['-c', ' && '.join(commands)],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={f"/{PROJECT_BASE}": pvolume}
    )


@dsl.pipeline(
    name='Fashion MNIST Training Pipeline',
    description='Fashion MNIST Training Pipeline to be executed on KubeFlow.'
)
def training_pipeline(image: str = 'dcavanau/kubeflow-mnist',
                      repo_url: str = 'https://61acc7bc6d89fb89dffb2c7e2142adffef6b13f1:x-oauth-basic@github.com/dcavanau/kubeflow-mnist.git',
                      data_dir: str = '/workspace',
                      model_version: int = 2):

    _git_clone = git_clone_op(repo_url=repo_url)

    _preprocess_data = preprocess_op(image=image,
                                    pvolume=_git_clone.pvolume,
                                    data_dir=data_dir)

    _training_and_eval = train_and_eval_op(image=image,
                                           pvolume=_preprocess_data.pvolume,
                                           data_dir=data_dir,
                                           model_path=data_dir,
                                           model_name='kubeflow-mnist',
                                           model_version=model_version)

    _packaging = packaging(image=image,
                           pvolume=_training_and_eval.pvolume,
                           model_path=data_dir,
                           model_name='kubeflow-mnist',
                           model_version=model_version)


    _publish = publish(pvolume=_packaging.pvolume)


if __name__ == '__main__':
    import kfp.compiler as compiler

    compiler.Compiler().compile(training_pipeline, __file__ + '.tar.gz')
