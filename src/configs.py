# -*- coding: utf-8 -*-
# @Time    : 2020/9/1 18:55
# @Author  : LIU YI

import numpy as np
import pandas as pd
import math
import copy

from matplotlib import pyplot as plt


def allocate_data(dataset_size, usr_num, enable_random = False):
    if not enable_random:
        return [dataset_size / usr_num for i in range(usr_num)]
    else:
        return [dataset_size / usr_num for i in range(usr_num)]

def generate_exec_time(usr_num, lo, hi) :
    '''
    [16.30823133 13.17435763 13.88474983 13.89481488 11.8644529  12.50021156
     17.04035062  6.01753793 11.00694407 11.4797031  10.17031539 15.63348328
     14.11423181 10.94848392 11.12879503  7.86005138  9.72729131 10.67684692
     16.86756606 10.66194789]
    '''
    '''
    [10.29325962  9.58373717 11.17336321 13.10025392 12.28072808 12.34930587
     14.36761575  7.03886069 11.27945065 19.6871113  12.6369494  13.15345891
     11.78414753 20.8101589  11.41113623 11.98350933 15.68428675 11.69254895
     17.49058905  7.0972756 ]
    '''
    '''
    2:
    [11.24972646 12.33119952  6.09141171 17.42081243  7.11969324  9.9747579
     14.00864425  8.76413574  9.32614334  9.77297716 14.15436213 19.37662404
     12.62461818  9.14622366 14.11717496 10.7115209  12.44260851 16.02500366
     10.25638715 12.52707575]
    '''
    np.random.seed(2)
    return np.random.normal((hi + lo) / 2, 3, usr_num)

class Configs(object):

    def __init__(self):

        ## TODO For FL training
        self.data = 'mnist'
        self.task_repeat_time = 1
        self.rounds = 50
        self.sv_computation_round = [i * 10 - 1 for i in range(1, int(self.rounds / 10) + 1)]
        self.frac = 1
        self.user_num = 10
        self.unit = 5
        self.lo = 5
        self.hi = 20
        self.R = 3 * self.unit
        # self.R = 1
        self.baseline = 2 # 0 - random, 1 - sv only, 2 - sv + time
        sample_num = self.unit
        # self.exec_speed = np.array([10, 11, 14, 5, 7])
        self.exec_speed = generate_exec_time(usr_num=self.user_num, lo=self.lo, hi=self.hi)
        self.FL_LR = 0.005
        self.model = 'cnn'
        self.iid = 0   # 0 = non-iid; 1 = iid
        self.unequal = 1   # 0 = equal; 1 = unequal
        self.gpu = 0   # 0 = CPU; 1 = GPU

        self.select = True
        self.selected_client = 0

        self.aggregation = 'avg'   # 'sv' or 'avg'


        # TODO for Fderated Env

        self.remove_client_index = None


        if self.data == 'cifar':
            # self.data_size = np.array([40, 38, 32, 46, 44]) * 250
            self.data_size = np.array([10000, 10000, 10000, 10000, 10000])
            self.batch_size = 10
            theta_num = 62006
            self.D = (self.data_size / 10) * (32 * (theta_num + 10 * (3 * 32 * 32))) / 1e9
        elif self.data == 'cifar100':
            self.data_size = np.array([40, 38, 32, 46, 44]) * 250
            self.batch_size = 10
            theta_num = 69656
            self.D = (self.data_size / 10) * (32 * (theta_num + 10 * (3 * 32 * 32))) / 1e9
        else:
            # self.data_size = np.array([12000, 10000, 8000, 14000, 16000])
            # self.data_size = np.array([6000, 12000, 12000, 14000, 16000])
            # self.data_size = np.array([24000, 9000, 9000, 9000, 9000])
            self.data_size = np.array(allocate_data(60000, usr_num=self.user_num))
            self.batch_size = 100
            theta_num = 21840
            self.D = (self.data_size / 10) * (32 * (theta_num + 10 * 28 * 28)) / 1e9

        self.data_size_original = copy.copy(self.data_size)



        self.frequency = np.array([1.4359949, 1.52592623, 1.04966248, 1.33532239, 1.7203678])

        if self.remove_client_index!=None:
            self.user_num = self.user_num-1
            self.data_size = np.delete(self.data_size, self.remove_client_index)
            self.frequency = np.delete(self.frequency, self.remove_client_index)
            self.D = np.delete(self.D, self.remove_client_index)


        self.C = 20
        self.alpha = 0.1
        self.local_epoch_range = 10

        self.performance = ['loss', 'acc']
        self.performance = self.performance[1]

        if self.performance == 'acc':
            if self.data == 'cifar100':
                self.lamda = 2000
            else:
                self.lamda = 1000    # todo changed for 10 rounds
        else:
            self.lamda = 4


        ## TODO For RL training

        self.EP_MAX = 2000
        self.S_DIM = self.user_num+1  # TODO add history later
        self.A_DIM = self.user_num
        self.BATCH = self.rounds  # TODO change round
        self.A_UPDATE_STEPS = 5
        self.C_UPDATE_STEPS = 5
        self.HAVE_TRAIN = False

        self.dec = 0.3
        self.A_LR = 0.00003  # todo  learning rate influence tendency
        self.C_LR = 0.00003
        self.GAMMA = 0.95
        # self.action_space = np.zeros((self.user_num, self.local_epoch_range))

        ## TODO For RL inference
        self.infer_round = 50


        ## TODO For myopia greedy

        self.myopia_frac = 1
        self.myopia_max_epoch = 1
#
#         self.lamda = 500
#
#         self.his_len = 5
#         self.info_num = 3
#
#
#
#         if self.data == 'cifar':
#             theta_num = 62006
#             data_size = np.array([40, 38, 32, 46, 44]) * 250 * 0.8
#
#         else:
#             theta_num = 21840
#             if self.user_num == 5:
#                 data_size = np.array([10000, 12000, 14000, 8000, 16000]) * 0.8
#             else:
#                 data_size = pd.read_csv('Multi_client_data/'+str(self.user_num)+'mnist.csv')
#                 data_size = np.array(data_size['data_size'].tolist())
#
#         self.D = (data_size / 10) * (32 * (theta_num + 10 * 28 * 28)) / 1e9
#         self.alpha = 0.1
#         self.tau = 1
#         self.C = 20
#         self.communication_time = np.random.uniform(low=10, high=20, size=self.user_num)
#
#         self.BATCH = 5   #todo
#         self.A_UPDATE_STEPS = 5  # origin:5
#         self.C_UPDATE_STEPS = 5
#         self.HAVE_TRAIN = False
#         self.A_LR = 0.00001  # origin:0.00003
#         self.C_LR = 0.00001
#         self.GAMMA = 0.95  # origin: 0.95
#         self.dec = 0.3
#
#         self.EP_MAX = 1000  #todo
#         self.EP_MAX_pre_train = 1000
#
#         self.EP_LEN = 100
#
#
#         if self.user_num == 5:
#             self.delta_max = np.array([1.4359949, 1.02592623, 1.54966248, 1.43532239, 1.4203678])
#
#             Loss = pd.read_csv('loss_mnist_500.csv')
#             Loss = Loss.to_dict()
#             Loss = Loss['1']
#             loss_list = []
#             for i in Loss:
#                 loss_list.append(Loss[i])
#
#
#             self.loss_list = copy.copy(loss_list)
#             num = len(loss_list)
#             buffer = 0
#             profit_increase = []
#             for i in range(0, num):
#                 loss_list[i] = -math.log(loss_list[i])
#
#             for one in loss_list:
#                 profit_increase.append(one - buffer)
#                 buffer = one
#
#             self.acc_increase_list = profit_increase
#
#
#
#         else:
#             self.delta_max = np.random.uniform(low=1, high=2, size=self.user_num)
#             data_info = pd.read_csv('Multi_client_data/' + str(self.user_num) + 'user_' + 'mnist' + '_1_0.005.csv')
#             accuracy_list = data_info['loss'].tolist()
#             num = len(accuracy_list)
#
#             self.loss_list = copy.copy(accuracy_list)
#             for i in range(0, num):
#                 accuracy_list[i] = -math.log(accuracy_list[i])
#
#             buffer = -math.log(3)
#
#             self.acc_increase_list = []
#             for one in accuracy_list:
#                 self.acc_increase_list.append(one - buffer)
#                 buffer = one
#
#         self.amplifier_baseline = np.max(self.delta_max * self.tau * self.C * self.D * self.alpha)
#         self.amplifier_hrl = np.sum(self.delta_max * self.tau * self.C * self.D * self.alpha)
#         self.reducer_baseline = 1
#         self.reducer_hrl = 100
#
#         reducer_pretrain_dict = {
#             5: 1000,
#             10: 500,
#             20: 50,
#             30: 50,
#             40: 50,
#             40: 50,
#             50: 50
#         }
#         self.reducer_pretrain = reducer_pretrain_dict[self.user_num]
#
#
if __name__ == '__main__':
    temp = generate_exec_time(usr_num=20, lo=5, hi=20)
    print(temp)
    temp2 = allocate_data(60000, usr_num=20)
    print(temp2)