tau=0.5
alpha=0.5 
for data in Cora Citeseer Pubmed ogbn-arxiv
do
  expname=$data'bs4096score2_tau='$tau"alpah="alpha
  dataset=$data
  CUDA_VISIBLE_DEVICES=5 python main.py --alpha $alpha --num_epoch 200 --score_type score2 --tau $tau --reinit True --dataset $dataset --keep_ratio 0.95 --batch_size 4096 --logdir log/$expname > log/$expname.log 2>&1 &
done
