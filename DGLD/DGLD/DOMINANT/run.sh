dataset=Cora
expname=$dataset"_DOMINANT"

if [ ! -d log  ];then
  mkdir log
  echo mkdir log
else
  echo dir exist
fi
# python main.py --dataset $dataset --logdir log/$expname > log/$expname.log
nohup python main.py --dataset $dataset --logdir log/$expname > log/$expname.log 2>&1 &