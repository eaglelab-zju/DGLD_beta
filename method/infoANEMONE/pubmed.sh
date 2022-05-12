for data in Pubmed 
do
  for tau in 0.95 0.9 0.8 0.85 
  do
    expname=$data'tr50_sela50CoLAInfos=keep_ratio'$tau
    dataset=$data
    CUDA_VISIBLE_DEVICES=3 python main.py --num_epoch 50 --selflabeling_epcohs 50 --dataset $dataset --keep_ratio $tau --batch_size 2048 --logdir log/$expname > log/$expname.log 
    # nohup python main.py --dataset $dataset --logdir log/$expname > log/$expname.log 2>&1 &
  done
done
