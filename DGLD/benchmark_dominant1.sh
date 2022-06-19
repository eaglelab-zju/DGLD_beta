if [ ! -d log  ];then
  mkdir log
  echo mkdir log
else
  echo log dir exist
fi

seed=4096
# for data in Cora Citeseer Pubmed ogbn-arxiv ACM Flickr BlogCatalog
for data in ACM
do
  expname=$data'_DOMINANT1'
  echo ${expname}
  dataset=$data
  CUDA_VISIBLE_DEVICES=5 python main_dominant1.py --dataset $dataset --device 0 --seed $seed --logdir log/$expname > log/$expname.log 2>&1
done



