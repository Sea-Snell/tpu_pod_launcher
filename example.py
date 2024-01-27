from tpu_pod_launcher import TPUPodClient, TPUPodProject, create_cli

SETUP_SCRIPT = """\
cd ~/
# install basics
apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    apt-utils \
    curl \
    git \
    vim \
    wget \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install miniforge
rm -rf ~/Miniconda3-py39_4.12.0-Linux-x86_64.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -P ~/
bash ~/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b

# install dependencies
source ~/miniconda3/bin/activate
conda init bash
cd ~/llama_train
python -m pip install -e .
python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# clean up
cd ~/
rm -rf ~/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
""".strip()

CHECK_DEVICES = r"""
source ~/miniconda3/bin/activate
python -c "import jax; print(jax.devices())"
""".strip()

def setup(project: TPUPodProject, verbose: bool=False):
    project.copy(verbose=verbose)
    project.ssh(SETUP_SCRIPT, verbose=verbose)
    project.ssh('mkdir ~/.config/', verbose=verbose)
    project.ssh('mkdir ~/.config/gcloud/', verbose=verbose)
    project.scp('/home/csnell/.config/gcloud/my_gcs_key.json', '~/.config/gcloud/', verbose=verbose)

def check_devices(project: TPUPodProject, verbose: bool=False):
    project.ssh(CHECK_DEVICES, verbose=verbose)

def debug(project: TPUPodProject, verbose: bool=False):
    import IPython; IPython.embed()

if __name__ == "__main__":
    projects = {
        'my_project': TPUPodProject(
            client=TPUPodClient(
                tpu_project='my_project',
                tpu_zone='europe-west4-a',
                user='charliesnell',
                key_path='/home/csnell/.ssh/general_key',
            ),
            tpu_name='your-node-id',
            copy_dirs=[('/home/csnell/llama_train/', '~/llama_train')],
            working_dir='~/llama_train',
            copy_excludes=['.git', '__pycache__'],
            kill_commands=['pkill -9 python'],
        )
    }

    create_cli(projects, setup, {'check_devices': check_devices, 'debug': debug})
