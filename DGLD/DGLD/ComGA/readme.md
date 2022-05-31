# ComGA
test datasets on 1080ti:'BlogCatalog','Flickr', 'cora','citeseer'
## Usage
ls
- modify dataset in run.sh,then run:
```shell
bash run.sh
```
log will save to `log` dir

- or you can run python script directly
```shell
python main.py --dataset Cora --logdir log/Cora_DOMINANT > log/Cora_DOMINANT.log
```