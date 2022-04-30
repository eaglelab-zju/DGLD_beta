dataset=BlogCatalog
expname=$dataset"_ComGA"

if [ ! -d log  ];then
  mkdir log
  echo mkdir log
else
  echo dir exist
fi
nohup python main.py --dataset $dataset --logdir log/$expname > log/$expname.log 2>&1 &
#test
# python main.py --dataset BlogCatalog --logdir log/BlogCatalog_ComGA > log/BlogCatalog_ComGA.log
