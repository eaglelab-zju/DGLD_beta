tau=0.3
alpha=0.8
for data in Cora Citeseer Pubmed ogbn-arxiv ACM
do
  expname=$data'bs4096info_tau='$tau"alpah="$alpha
  dataset=$data
  CUDA_VISIBLE_DEVICES=2 python main.py --alpha $alpha --score_type scorelossfixbatch --tau $tau --reinit True --dataset $dataset --keep_ratio 0.95 --batch_size 4096 --logdir log/$expname > log/$expname.log 2>&1 &
done