tau=0.2
alpha=0.8
loss='infonce'
batch=4096
for data in Cora Citeseer Pubmed ogbn-arxiv ACM
do
  expname=$data'colabs'$batch$loss'_tau='$tau"alpah="$alpha
  dataset=$data
  CUDA_VISIBLE_DEVICES=1 python main.py --loss_type $loss  --run 1 --alpha $alpha --score_type scorelossfixbatch --tau $tau --reinit True --dataset $dataset --keep_ratio 0.95 --batch_size $batch --logdir log/$expname > log/$expname.log 2>&1 &
done
