# 开始计时
start=$(date +%s)

expname=pubmed_info_raw_bs4096_exp0
dataset=Pubmed

if [ ! -d log  ];then
  mkdir log
  echo mkdir log
else
  echo dir exist
fi
 CUDA_VISIBLE_DEVICES=2 python main.py --num_epoch 100 --batch_size 4096 --dataset $dataset
# nohup python main.py --dataset $dataset --logdir log/$expname > log/$expname.log 2>&1 &

# 结束计时
end=$(date +%s)
# 计算耗时并输出
## 总预测时长（秒）
take=$(( end - start ))
echo "总预测时长: ${take} s"