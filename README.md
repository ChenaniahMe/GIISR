# GIISR

## About the code of the proposed method GIISR

This is the code for the AAAI 2021 Paper: [Graph Neural Networks with Intra- and
Inter-session Information for Session-based Recommendation] . We implemente the proposed method in **Pytorch**.

####(1) Description of datasets

We evaluate our method on two representative real-world datasets, i.e. Yoochoose and Diginetica. 

- YOOCHOOSE: <http://2015.recsyschallenge.com/challenge.html>

- DIGINETICA: <http://cikm2016.cs.iupui.edu/cikm-cup>

we use the sessions with length larger than 1 and the items appearing greater than 4 times in all datasets. we adopt the common fractions 1/64 and 1/4 for Yoochoose
dataset and Diginetica dataset .

####(2) Usage of the code

The diginetica datasets are included in the folder GIISR/datasets/diginetica, which can be used to test code of the proposed method.

If you need to use raw data to test the proposed method, you need first to run the python file  `preprocess.py` to preprocess the data. 

For example: `python preprocess.py --dataset=diginetica`

```bash
usage: preprocess.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  dataset name: diginetica/yoochoose/
```

Then you can run the file `pytorch/main.py` to train the model.

For example: `python main.py --dataset=diginetica`

You can use the corresponding parameters in the paper to verify our model

```bash
usage: main.py [-h] [--dataset DATASET] [--batchSize BATCHSIZE]
                [--metho METHOD] [--distance --DISTANCE]
               [--k K]
               [--hiddenSize HIDDENSIZE] [--epoch EPOCH] [--lr LR]
               [--lr_dc LR_DC] [--lr_dc_step LR_DC_STEP] [--l2 L2]
               [--step STEP] [--patience PATIENCE] [--nonhybrid]
               [--validation] [--valid_portion VALID_PORTION]
              

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset name:
                        diginetica/yoochoose1_4/yoochoose1_64/sample
  --batchSize BATCHSIZE
                        input batch size
  --hiddenSize HIDDENSIZE
                        hidden state size
  --epoch EPOCH         the number of epochs to train for 
  --k K                 the number of inter-session for the second stage
  --method METHOD       the different method, include GIISR(new2_mixed_posneg_br_one) and its variant
  --distance DISTANCE   the distance between inter-session for analysis on the Similarity Metrics of the Inter-session Graph
  --lr LR               learning rate
  --lr_dc LR_DC         learning rate decay rate
  --lr_dc_step LR_DC_STEP
                        the number of epochs after which the learning rate
                        decay
  --l2 L2               l2 penalty
  --step STEP           gnn propogation steps
  --patience PATIENCE   the number of epoch to wait before early stop
  --nonhybrid           only use the global preference to predict
  --validation          validation
  --valid_portion VALID_PORTION
                        split the portion of training set as validation set
```
####(3) The descripation of mode with code
In the first stage, the GCN in our work can initially model and extract the relation between items, which provides rich information for subsequent feature learning. 
```
input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
inputs = torch.cat([input_in, input_out], 2)
```
we present a Modified Residual Gated Neural Networks (MRGNN) to further extract and enhance this information for the next stage. 
```
w_za = F.linear(inputs, self.w_z, self.wb_z)
w_oa = F.linear(inputs, self.w_o, self.wb_o)
u_zv = F.linear(hidden, self.u_z, self.ub_z)
u_ov = F.linear(hidden, self.u_o, self.ub_o)
prelu = torch.nn.PReLU().cuda()
zs = prelu(w_za + u_zv)
vtb = prelu(w_oa +u_ov)
hy = zs*hidden + zs*vtb

w_za2 = F.linear(inputs, self.w_z2, self.wb_z2)
w_oa2 = F.linear(inputs, self.w_o2, self.wb_o2)
u_zv2 = F.linear(hy, self.u_z2, self.ub_z2)
u_ov2 = F.linear(hy, self.u_o2, self.ub_o2)
prelu = torch.nn.PReLU().cuda()
zs2 = prelu(w_za2 + u_zv2)
vtb2 = prelu(w_oa2 +u_ov2)
hy = zs2 * hy + zs2 * vtb2 + hy
```
In the second stage, we construct an inter-session graph which also contains the intra-session relations fused from the first stage to aggregate the information between different sessions. 
```
hy_flatten = hy.reshape(hy.shape[0], hy.shape[1] * hy.shape[2]).cuda()
hy_dis = cdist(hy_flatten.cpu().detach(), hy_flatten.cpu().detach(), metric='euclidean')
hy_dis = MinMaxScaler().fit(hy_dis).transform(hy_dis).T
hy_dis = torch.tensor(hy_dis, dtype=torch.float32)
hy_dis = hy_dis.pow(2)
# obtain positive k
idx_k_pos = np.argsort(hy_dis)[:, :opt.k]
# obtatin negative k
idx_k_neg = np.argsort(hy_dis)[:, -opt.k:]

# obatin positive A matrix
idx_len_pos = idx_k_pos.shape[0]
hy_k_pos = hy_dis[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos]
hy_i_k_pos = hy_k_pos[:, -1]
hy_j_k_pos = hy_k_pos[idx_k_pos, -1]
mole_pos = -hy_k_pos
deno_pos = hy_j_k_pos * hy_i_k_pos.reshape(hy_i_k_pos.shape[0], 1)
hy_ij_pos = torch.exp(mole_pos / deno_pos)
A_ij_pos = torch.zeros(hy_ij_pos.shape[0], hy_ij_pos.shape[0])
A_ij_pos[np.arange(0, idx_len_pos, 1).reshape(idx_len_pos, 1), idx_k_pos] = hy_ij_pos

hy_matmul_pos = torch.matmul(A_ij_pos.cuda(), hy_flatten)
hy_pos = hy_matmul_pos.reshape(hy.shape[0], hy.shape[1], hy.shape[2])

hy_neg = np.array([1])
hy = hy_pos
```
python main.py --dataset=diginetica --method=new2_mixed_posneg_br_one --batchSize=100 --k=11  --method_net_last=orginal --method_net_last_n1=last_n1_orginal

####(4) The experiment results
In the code, the corresponding experiment results can be saved in GIISR/logs/, GIISR/logs_loss, GIISR/logs_time 

## Requirements

- Python 3
- PyTorch 1.3.1



