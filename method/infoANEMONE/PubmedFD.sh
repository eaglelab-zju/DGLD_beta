for data in Pubmed 
do
  tau=0.2
  for epoch in 15 25 30 40 50 100
  do
    expname=$data'FDCoLAInfoNeg-Pos'$tau'epoch='$epoch
    dataset=$data
    CUDA_VISIBLE_DEVICES=3 python main.py --dataset $dataset --tau $tau --batch_size 2048 --num_epoch $epoch  --logdir log/$expname > log/$expname.log 
    # nohup python main.py --dataset $dataset --logdir log/$expname > log/$expname.log 2>&1 &
  done
done
#Flikcr BlogCatalog Cora  Citeseer