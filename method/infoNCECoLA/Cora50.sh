for data in Cora  Citeseer Pubmed
do
  for tau in 0.5 0.3
  do
    expname=$data'Pos_tr100_sela100epochCoLA_=tau'$tau
    dataset=$data
    CUDA_VISIBLE_DEVICES=1 python main.py --tau $tau --reinit True --num_epoch 100 --selflabeling_epcohs 100 --dataset $dataset --keep_ratio 0.95 --batch_size 2048 --logdir log/$expname > log/$expname.log 
    # nohup python main.py --dataset $dataset --logdir log/$expname > log/$expname.log 2>&1 &
  done
done
