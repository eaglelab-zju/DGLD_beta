for data in Cora Citeseer Pubmed ACM ogbn-arxiv
do
  for tau in 0.5
  do
    expname=$data'bs4096score2_=tau'$tau
    dataset=$data
    CUDA_VISIBLE_DEVICES=0 python main.py --score_type score2 --tau $tau --reinit True --dataset $dataset --keep_ratio 0.95 --batch_size 4096 --logdir log/$expname > log/$expname.log 
  done
done
