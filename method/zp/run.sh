expname=cora_info_posneg100_fix
dataset=Cora

if [ ! -d log  ];then
  mkdir log
  echo mkdir log
else
  echo dir exist
fi
 CUDA_VISIBLE_DEVICES=3 python main.py --num_epoch 100 --batch_size 300 --tau 1.0 --dataset $dataset --logdir log/$expname > log/$expname.log
# nohup python main.py --dataset $dataset --logdir log/$expname > log/$expname.log 2>&1 &
