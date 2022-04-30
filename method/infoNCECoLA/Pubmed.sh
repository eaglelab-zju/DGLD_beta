for data in Pubmed 
do
  for tau in 0.02 0.05 0.1 0.2 0.3 0.5 0.8 1.0 2.0
  do
    expname=$data'CoLAInfoPos'$tau
    dataset=$data
    CUDA_VISIBLE_DEVICES=4 python main.py --dataset $dataset --tau $tau --batch_size 2048 --num_epoch 100 --logdir log/$expname > log/$expname.log 
    # nohup python main.py --dataset $dataset --logdir log/$expname > log/$expname.log 2>&1 &
  done
done
#Flikcr BlogCatalog Cora  Citeseer