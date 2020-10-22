#!/usr/bin/env python36
# -*- coding: utf-8 -*-


import argparse
import pickle
import warnings
import time
from utils import build_graph, Data, split_validation
from model import *
# from model2 import *
from parsers import opt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(opt)
warnings.filterwarnings("ignore")
def main():
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
    # all_train_seq = pickle.load(open('../datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    # g = build_graph(all_train_seq)
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    # del all_train_seq, g
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    else:
        n_node = 310

    model = trans_to_cuda(SessionGraph(opt, n_node))
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        start_paper = time.time()
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        result = '\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1])
        print(result)
        # file_path = "../logs/"+opt.dataset+'batch'+str(opt.batchSize)+opt.method+str(opt.k)+str(opt.nonhybrid)+opt.method_net_last+opt.method_net_last_n1+".txt"
        file_path = "../logs/" + opt.dataset + 'batch' + str(opt.batchSize) + opt.method + str(opt.k)  + opt.distance + ".txt"
        with open(file_path, "a") as f:
            f.write(str(result))
            f.write('\n')

        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
        end_paper = time.time()
        time_paper=end_paper-start_paper
        print("each epoch all time->", end_paper-start_paper)
        file_path_time = "../logs_time/" + opt.dataset + 'batch' + str(opt.batchSize) + opt.method + str(
            opt.k) + opt.distance + ".txt"
        with open(file_path_time, "a") as f:
            f.write(str(time_paper))
            f.write('\n')
    file_path_two = "../logs_loss/" + opt.dataset + 'batch' + str(opt.batchSize) + opt.method + str(
        opt.k) + opt.distance + ".txt"
    with open(file_path_two, "a") as f:
        f.write('\n')
    file_path_three = "../logs_loss/" + opt.dataset + 'batch' + str(opt.batchSize) + opt.method + str(
        opt.k) + opt.distance + "all" + ".txt"
    with open(file_path_three, "a") as f:
        f.write('\n')
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))
    # file_path_time = "../logs/" + "time"+opt.dataset + 'batch' + str(opt.batchSize) + opt.method + str(
    #     opt.k) + opt.distance + ".txt"
    # with open(file_path_time, "a") as f:
    #     f.write(str(end))
    #     f.write('\n')

if __name__ == '__main__':
    main()
