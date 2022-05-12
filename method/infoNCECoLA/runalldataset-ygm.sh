for data in Cora,Citeseer,Pubmed,ACM,ogbn-arxiv 
do
  expname=$data'bs4096loss_log()_divide_nas_scoreinfoCoLA_=tau'$tau
  dataset=$data
  CUDA_VISIBLE_DEVICES=3 python main.py --tau $tau --reinit True --dataset $dataset --keep_ratio 0.95 --batch_size 4096 --logdir log/$expname > log/$expname.log 
done
