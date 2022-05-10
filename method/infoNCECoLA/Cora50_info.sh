for data in Cora Citeseer ACM Pubmed ogbn-arxiv
do
  for tau in 0.5
  do
    expname=$data'bs4096info_=tau'$tau
    dataset=$data
    CUDA_VISIBLE_DEVICES=5 python main.py --score_type scorelossfixbatch --tau $tau --reinit True --dataset $dataset --keep_ratio 0.95 --batch_size 4096 --logdir log/$expname > log/$expname.log 
  done
done
