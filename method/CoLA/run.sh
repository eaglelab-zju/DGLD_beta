expname=Cora_CoLA_global_adj_exp2
dataset=Cora

if [ ! -d log  ];then
  mkdir log
  echo mkdir log
else
  echo dir exist
fi
 CUDA_VISIBLE_DEVICES=3 python main.py --dataset $dataset --logdir log/$expname > log/$expname.log
# nohup python main.py --dataset $dataset --logdir log/$expname > log/$expname.log 2>&1 &
