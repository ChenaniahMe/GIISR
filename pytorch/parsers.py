import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/')
parser.add_argument('--method', default='new2_mixed_posneg_br_one', help='the method for different model')
parser.add_argument('--method_net_last', default='last_modify', help='the method from original or change')
parser.add_argument('--distance', default='euclidean', help='the distance between inter-session')
parser.add_argument('--method_net_last_n1', default='last_n1_modify', help='the method from original or change')
parser.add_argument('--method_addneg_mix', default='original', help='the method from original or change')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--k', type=int, default=15, help='the number of intra-session for two-stage')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()