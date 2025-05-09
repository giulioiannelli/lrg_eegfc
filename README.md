# lrg_eegfc
## installation
### conda environment
#### export
The following command is an utility for keeping the `environment.yml` correctly updated.
```
conda env export -n myenv --from-history | sed '/^prefix:/d' > environment.yml
```