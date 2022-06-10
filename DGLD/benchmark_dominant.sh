if [ ! -d log  ];then
  mkdir log
  echo mkdir log
else
  echo log dir exist
fi

for data in Cora Citeseer Pubmed ogbn-arxiv ACM Flickr BlogCatalog
do
  expname=$data'_DOMINANT'
  echo ${expname}
  dataset=$data
  CUDA_VISIBLE_DEVICES=2 PYTHONHASHSEED=1024 python main_dominant.py --dataset $dataset --device 0 --seed 1024 --logdir log/$expname > log/$expname.log 2>&1
done



