# lrg_eegfc

## installation
It requires `conda` to be installed locally in path `$HOME/anaconda3/`. Start cloning and initing the submodules:
```
git clone https://github.com/giulioiannelli/lrg_eegfc
git submodule update --init --recursive
```
To update the `` submodule run
```
cd lrgsglib-public
git fetch origin
git merge origin/main
```
### prepare the conda environment
Create a custom conda environment, e.g. `lapbrain`,
```
cd lrgsglib-public
conda env create -n lapbrain -f lrgsgenv.yml
```
### configure with make
```
cd lrgsglib-public
make CONDA_ENV_NAME=lapbrain
```
### install `lrgsglib` in editable mode
```
pip install --editable .
```
### conda utils
#### export
The following command is an utility for keeping the `lapbrain.yml` correctly updated.
```
conda env export -n lapbrain --no-builds \
  | sed '/^prefix:/d' \
  | sed '/^  - pip:/,$d' \
  > lapbrain.yml
```