import textwrap
from launcher import TPUPodClient, TPUPodProject

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
conda env create -f environment.yml
conda activate JaxSeq2
python -m pip install --upgrade pip && python -m pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# clean up
rm -rf ~/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
""".strip()

if __name__ == "__main__":
    client = TPUPodClient(
        tpu_project='civic-boulder-204700',
        tpu_zone='us-east1-d',
        user='charliesnell',
        key_path='/home/csnell/.ssh/general_key',
    )
    project = TPUPodProject(
        client=client,
        tpu_name='small-pod',
        copy_dirs=[('/home/csnell/JaxSeq2_experimental/', '~/JaxSeq2_experimental')],
        working_dir='~/JaxSeq2_experimental',
        copy_excludes=['.git', '__pycache__'],
        kill_commands=['pkill -9 python'],
    )

    # project.ssh(SETUP_SCRIPT, verbose=True)
    # project.copy(verbose=True)
    project.ssh('pwd', verbose=True)
    # project.ssh("rm -rf ~/JaxSeq2_experimental", verbose=True)
