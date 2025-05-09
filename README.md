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
Create the environment `lrgsgenv` as from file `lrgsgenv.yml` in 
```
cd lrgsglib-public
conda env create -f lrgsgenv.yml
```
Then if you want to continue working on a custom conda environment, e.g. `lapbrain`, do
```
conda env update -n lapbrain -f lrgsgenv.yml
```
### configure with make
```
cd lrgsglib-public
make all
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