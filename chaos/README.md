# Create environment

## Install pyenv
<!-- git clone https://github.com/pyenv/pyenv.git ~/.pyenv
git clone git://github.com/yyuu/pyenv-update.git ~/.pyenv/plugins/pyenv-update

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile 
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile 
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
source ~/.bashrc
source ~/.bash_profile

pyenv update 
pyenv install anaconda3-2020.02
pyenv local anaconda3-2020.02 -->

sudo apt purge python-pip python3-pip
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
git clone git://github.com/yyuu/pyenv-update.git ~/.pyenv/plugins/pyenv-update
edit .zshrc
+ export PYENV_ROOT="$HOME/.pyenv"
+ export PATH="$PYENV_ROOT/bin:$PATH"
+ eval "$(pyenv init -)"

source ~/.zshrc

sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl

pyenv update
pyenv install 3.7.7
python3 -m venv ~/vqrc 
source ~/vqrc/bin/activate


# NON-LINEAR

pip3 install numpy matplotlib sklearn 

# CHAOS
# Run ESN, HQRC
pip3 install psutil

# Run LSTM, GRU
## To install tensorflow
pip install tensorflow==1.15.0
sudo apt install libopenmpi-dev
pip3 install mpi4py

# Run parallel

<!-- # conda install tensorflow==1.14.0
# conda install tensorflow==2.2.0

## To install mpi4py
## conda install mpi4py -->



<!-- ## To install utils
pip3 install matplotlib sklearn psutil -->

<!-- ## To install mpi4y
sudo apt install libopenmpi-dev
pip install mpi4py -->



## Generate data
cd Data/Lorenz3D
python data_generation.py

cd Data/KuramotoSivashinskyGP64
python data_generation.py