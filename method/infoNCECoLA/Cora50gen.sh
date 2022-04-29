for data in Cora Pubmed
do
  for generative_loss_w in 100 10 1 50 40
  do
    expname=$data'gen——50epochCoLAInfoNeg-Posgenerative_loss_w='$generative_loss_w
    dataset=$data
    CUDA_VISIBLE_DEVICES=1 python main.py --generative_loss_w $generative_loss_w --num_epoch 50 --dataset $dataset --tau 0.5 --batch_size 2048 --logdir log/$expname > log/$expname.log 
    # nohup python main.py --dataset $dataset --logdir log/$expname > log/$expname.log 2>&1 &
  done
done
