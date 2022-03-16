expname=Cora_norm_None_CoLA 
dataset=Cora
nohup python main.py --dataset $dataset --logdir log/$expname > log/$expname.log 2>&1 &
