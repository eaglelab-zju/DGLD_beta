if [ ! -d all_log  ];then
  mkdir all_log
  echo mkdir all_log
else
  echo dir exist
fi

datasets=(Cora Citeseer Pubmed ACM ogbn-arxiv)  #遍历5个数据集
gpu_id=3  #gpu id
exp_id=8_1  #实验次数/编号 
#exp_id=4 对正负子图做node_shuffle数据增强
#exp_id=5 只对负子图做node_shuffle数据增强
#exp_id=6 只对负子图做random_mask数据增强
#exp_id=7 对正负子图做random_mask=0.5 数据增强  #公式（12）
#exp_id=8 对正负子图做random_mask=0.5 tau=0.3  数据增强 #公式（12）
#exp_id=8_0 对正负子图做random_mask=0.5 tau=0.3  数据增强 #公式（12）
#exp_id=8_1 对正负子图做random_mask=0.5 tau=0.3  数据增强 #公式（12）

aug_type='random_mask'  #数据增强类型
aug_ratio=0.5 #数据增强的drop等概率值
score_type='scorelossfixbatch' #公式（12）
tau=0.3 #tau


for dataset in ${datasets[@]}
do
  expname=$dataset'_info_bs4096_aug_exp'$exp_id
  dataset=$dataset
  CUDA_VISIBLE_DEVICES=${gpu_id} nohup python main.py --num_epoch 100 --batch_size 4096 --dataset ${dataset} --aug_type ${aug_type} --aug_ratio ${aug_ratio} --score_type ${score_type} --tau ${tau}> all_log/${expname}.log 2>&1 &
done
