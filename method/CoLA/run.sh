expname=Cora_norm_None_CoLA 
dataset=Cora

if [ ! -d log  ];then
  mkdir log
  echo mkdir log
else
  echo dir exist
fi
# python main.py --dataset $dataset --logdir log/$expname > log/$expname.log
nohup python main.py --dataset $dataset --logdir log/$expname > log/$expname.log 2>&1 &
