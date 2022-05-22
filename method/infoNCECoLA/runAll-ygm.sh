if [ ! -d all_log  ];then
  mkdir all_log
  echo mkdir all_log
else
  echo dir exist
fi

# datasets=(Cora Citeseer Pubmed ACM ogbn-arxiv)  #遍历5个数据集
datasets=(ogbn-arxiv ACM)
gpu_id=5  #gpu id
exp_id=1  #实验次数/编号 
# exp_id=0 原始的数据

# aug_type='random_mask'  #数据增强类型
# aug_ratio=0.5 #数据增强的drop等概率值
score_type='scorelossfixbatch' #公式（12）
tau=0.3 #tau

for dataset in ${datasets[@]}
do
  expname=$dataset'_info_bs4096_raw_exp'$exp_id
  dataset=$dataset
  CUDA_VISIBLE_DEVICES=${gpu_id} nohup python main.py --batch_size 4096 --dataset ${dataset} --score_type ${score_type} --tau ${tau}> all_log/${expname}.log 2>&1 &
done

