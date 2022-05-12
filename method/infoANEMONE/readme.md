# cora
## add gcn
loss_pool *0.8 + loss_gcn*0.2
mean(0.880)
## remove gcn
loss_pool
mean(0.83)
## 降低pool的weight
loss_pool*0.5 + loss_gcn*0.5

## 
nohup python main.py --num_epoch 50 --dataset Cora --batch_size 2048 --tau 0.5
### key_emb = torch.cat([pos_emb, neg_emb], dim=0)
concat——50epochCoLAInfoNeg 
mean(905)
### key_emb = neg_emb
mean(905)
### loss = loss_pool*0.5 + loss_gcn*0.5
mean(843)
### 原始特征做NCE

### mask掉邻居

## ps
### GT
0.982
### ps

### 固定ps
### 打印pseudo的质量


### scoring
neg_score = 0
structural anomaly score: 0.8105029971331769
attribute anomaly score: 0.8637998436278342
final anomaly score: 0.8371514203805056
属性异常检测效果变好了

CUDA_VISIBLE_DEVICES=3
但是最终multiround效果比较差

### add gcn
VALID==>epoch:: 72:: Average valid loss: 13.71
structural anomaly score: 0.7506906437320823
attribute anomaly score: 0.7810841803492311
final anomaly score: 0.7658874120406567

变差了。。。
### anchor 也过mlp
变差了。。。
### remove label noise on 分母 

### EMA