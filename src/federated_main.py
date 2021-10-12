#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import heapq
import os
import copy
import time
import json
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt        #matplotlib inline
from matplotlib.ticker import MaxNLocator
import csv

import torch
from tensorboardX import SummaryWriter
from RL_brain import PPO
from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, LeNet
from utils import get_dataset, average_weights, exp_details, sv_weights
import pandas as pd
import random
import threading
import itertools

from configs import Configs
from DNC_PPO import PPO
from itertools import product

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class Env(object):

    def __init__(self, configs):
        self.configs = configs
        self.data_size = configs.data_size
        self.frequency = configs.frequency
        self.C = configs.C
        self.lamda = configs.lamda
        self.seed = 0
        self.D = configs.D
        self.history_avg_price = np.zeros(self.configs.user_num)

        # load dataset and user groups
        self.args = args_parser()
        self.train_dataset, self.test_dataset, self.user_groups = get_dataset(self.args)
        np.save('user_groups_cifar.npy', self.user_groups)

        # read_dictionary = np.load('user_groups_normal_non_iid_1.npy', allow_pickle=True).item()
        # self.user_groups = read_dictionary

        # count = 0
        # for idx in read_dictionary[0]:
        #     if idx not in self.user_groups[0]:
        #         count += 1
        # if count != 0:
        #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # else:
        #     print("#############################################################")

        # maintain the original data idx of selected client
        self.user_groups_all = []
        self.user_groups_all = self.user_groups[self.configs.selected_client]
        print("The lenth of selected user_groups_all {} is {}.\n".format(self.user_groups_all, len(self.user_groups_all)))

    def reset_for_greedy(self):
        self.index = 0
        self.data_value = 0.001 * self.data_size
        self.unit_E = self.configs.frequency * self.configs.frequency * self.configs.C * self.configs.D * self.configs.alpha  # TODO
        self.bid = self.data_value + self.unit_E
        self.bid_ = np.zeros(self.configs.user_num)
        self.action_history = []
        # self.bid_min = 0.7 * self.bid

        # todo annotate these random seed if run greedy, save them when run DRL
        start_time = time.time()
        self.acc_list = []
        self.loss_list = []
        # define paths
        path_project = os.path.abspath('..')
        self.logger = SummaryWriter('../logs')

        self.args = args_parser()
        exp_details(self.args)

        if self.configs.gpu:
            # torch.cuda.set_device(self.args.gpu)
            # device = 'cuda' if args.gpu else 'cpu'

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        else:
            device = 'cpu'

        # load dataset and user groups
        self.train_dataset, self.test_dataset, self.user_groups = get_dataset(self.args)


        if self.configs.remove_client_index != None:
            self.user_groups.pop(self.configs.remove_client_index)

        # BUILD MODEL
        if self.args.model == 'cnn':
            # Convolutional neural netork
            if self.args.dataset == 'mnist':
                self.global_model = CNNMnist(args=self.args)
            elif self.args.dataset == 'fmnist':
                self.global_model = CNNFashion_Mnist(args=self.args)
            elif self.args.dataset == 'cifar':
                self.global_model = CNNCifar(args=self.args)
            elif self.args.dataset == 'cifar100':
                self.global_model = LeNet()

        elif self.args.model == 'mlp':
            # Multi-layer preceptron
            img_size = self.train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
                self.global_model = MLP(dim_in=len_in, dim_hidden=64,
                                        dim_out=self.args.num_classes)
        else:
            exit('Error: unrecognized model')

        # Set the model to train and send it to device.

        print(get_parameter_number(self.global_model))
        print('---------------------------------------------------------------------------------------')

        self.global_model.to(device)
        self.global_model.train()
        print(self.global_model)

        # copy weights
        global_weights = self.global_model.state_dict()

        # Training
        self.train_loss, self.train_accuracy = [], []
        self.test_loss, self.test_accuracy = [], []
        self.acc_before = 0
        self.loss_before = 300
        self.val_acc_list, self.net_list = [], []
        self.cv_loss, self.cv_acc = [], []
        self.print_every = 1
        val_loss_pre, counter = 0, 0

        return self.bid

    def reset(self):
        self.index = 0



        # todo annotate these random seed if run greedy, save them when run DRL
        # np.random.seed(self.seed)
        # torch.random.manual_seed(self.seed)
        # random.seed(self.seed)
        # torch.cuda.manual_seed_all(self.seed)
        # torch.cuda.manual_seed(self.seed)

        start_time = time.time()
        self.acc_list = []
        self.loss_list = []
        # define paths
        path_project = os.path.abspath('..')
        self.logger = SummaryWriter('../logs')

        self.args = args_parser()
        exp_details(self.args)

        if self.configs.gpu:
            # torch.cuda.set_device(self.args.gpu)
            # device = 'cuda' if args.gpu else 'cpu'

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        else:
            device = 'cpu'

        if self.configs.select == True:  # whether select partial data for training
            print("The lenth of previous selected self.user_groups {} is {}.".format(self.user_groups[self.configs.selected_client], len(self.user_groups[self.configs.selected_client])))

            self.user_groups[self.configs.selected_client] = np.random.choice(self.user_groups_all, int(0.5 * len(self.user_groups_all)),
                                                   replace=None)  # select 50% data from client for training

            print("The lenth of current self.user_groups_selected {} is {}:".format(self.user_groups[self.configs.selected_client], len(self.user_groups[self.configs.selected_client])))

        if self.configs.remove_client_index != None:
            self.user_groups.pop(self.configs.remove_client_index)

        # BUILD MODEL
        if self.args.model == 'cnn':
            # Convolutional neural netork
            if self.args.dataset == 'mnist':
                self.global_model = CNNMnist(args=self.args)
            elif self.args.dataset == 'fmnist':
                self.global_model = CNNFashion_Mnist(args=self.args)
            elif self.args.dataset == 'cifar':
                self.global_model = CNNCifar(args=self.args)
            elif self.args.dataset == 'cifar100':
                self.global_model = LeNet()

        elif self.args.model == 'mlp':
            # Multi-layer preceptron
            img_size = self.train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
                self.global_model = MLP(dim_in=len_in, dim_hidden=64,
                                   dim_out=self.args.num_classes)
        else:
            exit('Error: unrecognized model')

        # Set the model to train and send it to device.

        print(get_parameter_number(self.global_model))
        print('---------------------------------------------------------------------------------------')

        self.global_model.to(device)
        self.global_model.train()
        print(self.global_model)

        # copy weights
        global_weights = self.global_model.state_dict()

        # Training
        self.train_loss, self.train_accuracy = [], []
        self.test_loss, self.test_accuracy = [], []
        self.acc_before = 0
        self.loss_before = 300
        self.val_acc_list, self.net_list = [], []
        self.cv_loss, self.cv_acc = [], []
        self.print_every = 1
        val_loss_pre, counter = 0, 0



    # TODO   for multi- thread
    # def individual_train(self, idx):
    #     local_ep = self.local_ep_list[idx]
    #
    #     if local_ep != 0:
    #         local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
    #                                   idxs=self.user_groups[idx], logger=self.logger)
    #         w, loss = local_model.update_weights(
    #             model=copy.deepcopy(self.global_model), global_round=self.index, local_ep=local_ep)
    #         self.local_weights.append(copy.deepcopy(w))
    #         self.local_losses.append(copy.deepcopy(loss))

    def fake_step(self):

        weights_rounds, local_losses = [], []
        print(f'\n | Global Training Round : {self.index + 1} |\n')

        global_model_tep = copy.deepcopy(self.global_model)

        global_model_tep.train()

        idxs_users = list(self.user_groups.keys())

        # Local Training



        possible_epochs = list(range(1,self.configs.myopia_max_epoch+1))
        for epoch in possible_epochs:
            weights_users = []
            for idx in idxs_users:

                local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                          idxs=self.user_groups[idx], logger=self.logger)
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(self.global_model), global_round=self.index, local_ep=epoch)
                weights_users.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
            weights_rounds.append(copy.deepcopy(weights_users))

        possible_epochs = list(range(self.configs.myopia_max_epoch+1))
        loop_val = []
        for i in range(self.configs.user_num):
            loop_val.append(possible_epochs)

        result_book = pd.DataFrame([], columns=["action", "reward", "accuracy", "cost", "energy"], index=None)

        for i in product(*loop_val):
            if random.uniform(0, 1) > self.configs.myopia_frac:
                continue
            weights_tep = []
            action = list(i)
            for one in action:
                if one:
                    weights_tep.append(weights_rounds[one-1][action.index(one)])
            if weights_tep != []:
                global_weights = average_weights(weights_tep)
                global_model_tep = copy.deepcopy(self.global_model)
                global_model_tep.load_state_dict(global_weights)
                global_model_tep.eval()
                test_acc, test_loss = test_inference(self.args, global_model_tep, self.test_dataset)

                delta_acc = test_acc - self.acc_before
                delta_loss = self.loss_before - test_loss
                action = np.array(action)
                time_cmp = (action * self.D * self.C) / self.frequency
                time_global = np.max(time_cmp)

                action_cost = []
                for one in range(self.configs.user_num):
                    action_cost.append(self.configs.myopia_max_epoch)
                action_cost = np.array(action_cost)



                if self.configs.performance == 'acc':
                    delta_performance = delta_acc
                else:
                    delta_performance = delta_loss

                reward = (self.lamda * delta_performance - cost)/10 #TODO test for the existance of data importance

                print(action, reward)
                result_book = result_book.append([{'action': action, 'reward': reward, "accuracy": delta_acc, "cost": cost, "energy": E}])

        result_book.to_csv('Result_book_of_round_'+str(self.index)+'.csv', index=None)

        return result_book.sort_values('reward').iloc[-1]['action'], result_book.sort_values('reward').iloc[-1]['reward'], result_book.sort_values('reward').iloc[-1]['accuracy'], result_book.sort_values('reward').iloc[-1]['cost'], result_book.sort_values('reward').iloc[-1]['energy']

    def step(self, action, round):
        print('[STEP] Step Start')
        self.local_weights, self.local_losses = [], []
        print(f'\n | Global Training Round : {self.index + 1} |\n')
        self.global_model.train()
        idxs_users = np.array(list(self.user_groups.keys()))
        print("User index:", idxs_users)

        # TODO DRL
        action = 5 * action
        action = action.astype(int)
        self.local_ep_list = action

        for idx in idxs_users:
            local_ep = self.local_ep_list[list(idxs_users).index(idx)]
            print('[INNER INFO] Usr {} exec local update? {}'.format(idx, local_ep != 0))
            # local_ep always == 1, might be lead to an 1/0 judgement
            if local_ep != 0:
                local_model = LocalUpdate(args=self.args, dataset=self.train_dataset,
                                          idxs=self.user_groups[idx], logger=self.logger)
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(self.global_model), global_round=self.index, local_ep=local_ep)
                self.local_weights.append(copy.deepcopy(w))
                self.local_losses.append(copy.deepcopy(loss))
        # Shapley Value:
        acc_diff = {i:[] for i in range(self.configs.unit)}
        sv = []
        threshold = 12
        if self.configs.baseline != 0:
            print("[CRITIC] Start Shapley Value Part")
            # all permutations: i.e. 5 usr = 120
            perm_list = []
            perm_list += list(itertools.permutations(np.arange(self.configs.unit), self.configs.unit))
            print('[INFO] Permutations list: ', perm_list)
            print('[INFO] Permutations list length: ', len(perm_list))
            print('[INFO] Select first {} as sample'.format(self.configs.R))

            # TODO random select configs.R permutations
            empty_acc, empty_loss = test_inference(self.args, self.global_model, self.test_dataset)
            print('[INFO] Empty Accuracy {}'.format(empty_acc * 100))
            np.random.seed(int(time.time()))
            r_perm_index = np.random.choice([i for i in range(len(perm_list))], self.configs.R, replace = False)
            r_perm = []
            for i in r_perm_index:
                r_perm.append(perm_list[i])
            # r_perm = random.sample(perm_list, self.configs.R)
            print('[INFO] R random permutations {}'.format(r_perm))
            for i in r_perm:
                seq = list(i)
                # initialize
                queue = []
                accuracies = [empty_acc]
                loss = []
                queue_weights = []
                print('[INFO] Processing {}'.format(seq))
                for item in seq:
                    queue.append(item)
                    # print('[MID-INFO] Queue {}'.format(queue))
                    queue_weights.append(self.local_weights[item])
                    # print('[MID-INFO] Queue weights({}) {}'.format(len(queue_weights), '...'))
                    avg = average_weights(queue_weights)
                    self.global_model.load_state_dict(avg)
                    queue_acc, queue_loss = test_inference(self.args, self.global_model, self.test_dataset)
                    print('[SUB-INFO] {} contributes {}'.format(item, 100*(queue_acc - accuracies[-1])))
                    acc_diff[item].append(queue_acc - accuracies[-1])
                    accuracies.append(queue_acc)
                    loss.append(queue_loss)
            # Avg sv
            for i in range(self.configs.unit):
                sv.append(np.mean(acc_diff[i]))
            print('[INFO] SV({}): {}'.format(len(sv), sv))
            task_acc = 0
            task_loss = 0
        if self.configs.aggregation == 'sv':
            # TODO sv-based aggregation accuracy
            global_sv_weignts = sv_weights(self.local_weights, sv)
            self.global_model.load_state_dict(global_sv_weignts)
            sv_test_acc, sv_test_loss = test_inference(self.args, self.global_model, self.test_dataset)
            print('SV-based Aggregation Test Accuracy: {:.2f}% \n'.format(100 * sv_test_acc))
        elif self.configs.aggregation == 'avg':
            # TODO Avg aggregation accuracy
            global_weights = average_weights(self.local_weights)
            self.global_model.load_state_dict(global_weights)
            test_acc, test_loss = test_inference(self.args, self.global_model, self.test_dataset)
            task_acc, task_loss = test_acc, test_loss
            print('Avg Aggregation Test Accuracy: {:.2f}% \n'.format(100 * test_acc))
        print('[STEP] Step end')
        self.index += 1
        return sv, task_acc, task_loss


def fed_avg():
    configs = Configs()
    env = Env(configs)
    env.reset()
    data = pd.DataFrame([], columns=['action', 'reward', 'delta_accuracy', 'round_time', 'energy'])


    for one in range(configs.rounds):
        action = []
        for i in range(configs.user_num):
            action.append(1)
        action = np.array(action) / 5
        reward, next_bid, delta_accuracy, cost, round_time, int_action, energy = env.step(action)
        data = data.append([{'action': action, 'reward': reward, 'delta_accuracy': delta_accuracy,
                             'round_time': round_time, 'energy': energy, 'cost': cost}])
    data.to_csv('fed_avg1.csv', index=None)

def Greedy_myopia():
    configs = Configs()
    env = Env(configs)
    env.reset()
    data = pd.DataFrame([], columns=['action', 'reward', 'delta_accuracy', 'round_time', 'energy'])

    for one in range(configs.rounds):
        action, reward, delta_accuracy, cost, energy = env.fake_step()
        data = data.append([{'action': action, 'reward': reward, 'delta_accuracy': delta_accuracy, 'round_time': None, 'energy': energy, 'cost': cost}])
        action = np.array(action)/5
        reward, next_bid, delta_accuracy, cost, round_time, int_action, energy = env.step(action)
    data.to_csv('Greedy_myopia.csv', index=None)

def DRL_inference(agent_info):
    configs = Configs()
    env = Env(configs)
    ppo = PPO(configs.S_DIM, configs.A_DIM, configs.BATCH, configs.A_UPDATE_STEPS, configs.C_UPDATE_STEPS, True, agent_info)

    for remove_client_for_vcg in [0, 1, 3, 4]:

        recording = pd.DataFrame([], columns=['state history', 'action history', 'reward history', 'acc increase hisotry', 'time hisotry', 'energy history', 'social welfare', 'accuracy', 'time', 'energy'])


        for EP in range(50):
            cur_bid = env.reset()
            cur_state = np.append(cur_bid, env.index)

            state_list = []
            action_list = []
            reward_list = []
            performance_increase_list = []
            time_list = []
            cost_list = []
            energy_list = []
            for t in range(configs.rounds):
                print("Current State:", cur_state)
                action = ppo.choose_action(cur_state, configs.dec)
                action[remove_client_for_vcg] = 0

                while (np.floor(5 * action) == np.zeros(configs.user_num, )).all():
                    action = ppo.choose_action(cur_state, configs.dec)
                    action[remove_client_for_vcg] = 0

                print(action)
                reward, next_bid, delta_accuracy, cost, round_time, int_action, energy = env.step(action)

                cur_bid = next_bid
                next_state = np.append(next_bid, env.index)
                cur_state = next_state

                cost_list.append(cost)
                state_list.append(cur_state)
                action_list.append(int_action)
                reward_list.append(reward)
                performance_increase_list.append(delta_accuracy)
                time_list.append(round_time)
                energy_list.append(energy)

            recording = recording.append([{'state history': state_list, 'action history': action_list, 'reward history':reward_list, 'acc increase hisotry': performance_increase_list, 'time hisotry': time_list, 'energy history': energy_list, 'social welfare': np.sum(reward_list), 'accuracy': np.sum(performance_increase_list), 'time': np.sum(time_list), 'energy': np.sum(energy_list), 'cost_sum': np.sum(cost_list)}])
            recording.to_csv(agent_info+'_Remove_'+ str(remove_client_for_vcg) + '_Inference result.csv')

def DRL_train():

    configs = Configs()
    env = Env(configs)
    agent_info = str(configs.remove_client_index)+configs.data+'_'+configs.performance + time.strftime("%Y-%m-%d", time.localtime())
    # ppo = PPO(configs.S_DIM, configs.A_DIM, configs.BATCH, configs.A_UPDATE_STEPS, configs.C_UPDATE_STEPS, configs.HAVE_TRAIN, agent_info)
    ppo = PPO(configs.S_DIM, configs.A_DIM, configs.BATCH, configs.A_UPDATE_STEPS, configs.C_UPDATE_STEPS,
              True, '1cifar_acc2021-01-14')
    #todo num=0 2rounds on GPU; num=1 10rounds; num=2 20rounds of TestAcc; num=3 10Rounds test for data importance

    csvFile1 = open("remove"+str(configs.remove_client_index)+"_Result_summary(Continue)_" + str(configs.user_num) + "Client_"+configs.data+".csv", 'w', newline='')
    writer1 = csv.writer(csvFile1)

    accuracies = []
    costs = []
    round_times = []

    rewards = []
    closses = []
    alosses = []
    dec = configs.dec
    A_LR = configs.A_LR
    C_LR = configs.C_LR
    C_loss = pd.DataFrame(columns=['Episodes', 'C-loss'])

    for EP in range(configs.EP_MAX):
        cur_bid = env.reset()
        cur_state = np.append(cur_bid, 0)  #TODO  add index into state
        recording = []
        recording.append(cur_state)

        #  learning rate change for trade-off between exploit and explore
        if EP % 20 == 0:
            dec = dec * 0.95
            A_LR = A_LR * 0.85
            C_LR = C_LR * 0.85

        buffer_s = []
        buffer_a = []
        buffer_r = []
        sum_accuracy = 0
        sum_cost = 0
        sum_round_time = 0
        sum_reward = 0
        sum_action = 0
        sum_closs = 0
        sum_aloss = 0
        sum_energy = 0

        for t in range(configs.rounds):
            # local_ep_list = input('please input the local epoch list:')
            # local_ep_list = local_ep_list.split(',')
            # local_ep_list = [int(i) for i in local_ep_list]
            # action = local_ep_list
            print("Current State:", cur_state)

            action = ppo.choose_action(cur_state, configs.dec)
            while (np.floor(5*action) == np.zeros(configs.user_num,)).all():
                action = ppo.choose_action(cur_state, configs.dec)

            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(action)
            # while action == np.array([0,0,0,0,0]):
            #     action = ppo.choose_action(observation, configs.dec)
            reward, next_bid, delta_accuracy, cost, round_time, int_action, energy = env.step(action)

            # next_bid = cur_bid  # todo Fix biding, to be deleted after trial experiment

            sum_accuracy += delta_accuracy
            sum_cost += cost
            sum_round_time += round_time
            sum_reward += reward
            sum_action += action
            sum_energy += energy
            buffer_a.append(action.copy())
            buffer_r.append(reward)
            buffer_s.append(cur_state.reshape(-1, configs.S_DIM).copy())

            next_state = np.append(next_bid, t+1)
            recording.append(int_action)
            recording.append(reward)
            recording.append(next_state)

            print("Current State:", cur_state)
            print("Next State:", next_state)

            #  ppo.update()
            if (t+1) % configs.BATCH == 0:
                print("------------PPO UPDATED------------")
                discounted_r = np.zeros(len(buffer_r), 'float32')
                v_s = ppo.get_v(next_state.reshape(-1, configs.S_DIM))
                running_add = v_s

                for rd in reversed(range(len(buffer_r))):
                    running_add = running_add * configs.GAMMA + buffer_r[rd]
                    discounted_r[rd] = running_add

                discounted_r = discounted_r[np.newaxis, :]
                discounted_r = np.transpose(discounted_r)
                if configs.HAVE_TRAIN == False:
                    closs, aloss = ppo.update(np.vstack(buffer_s), np.vstack(buffer_a), discounted_r, configs.dec, configs.A_LR, configs.C_LR, EP+1)
                    sum_closs += closs
                    sum_aloss += aloss
                    C_loss = C_loss.append([{'Episodes': EP, 'C-loss': closs}])

            #TODO state transition
            cur_state = next_state
            print("################################# ROUND END #####################################")

        if (EP+1) % 1 == 0:
            print("------------------------------------------------------------------------")
            print('instant ep:', (EP+1))

            rewards.append(sum_reward * 10)
            # actions.append(sum_action / configs.rounds)
            closses.append(sum_closs / configs.rounds)
            alosses.append(sum_aloss / configs.rounds)
            accuracies.append(sum_accuracy)
            costs.append(sum_cost)
            round_times.append(sum_round_time)

            recording.append(sum_reward * 10)
            # recording.append(np.floor(5*(sum_action / configs.rounds)))
            recording.append(sum_closs / configs.rounds)
            recording.append(sum_aloss / configs.rounds)
            recording.append(sum_accuracy)
            recording.append(sum_cost)
            recording.append(sum_round_time)
            recording.append(sum_energy)
            writer1.writerow(recording)

            print("accumulated reward:", sum_reward * 10)
            # print("average action:", sum_action / configs.rounds)
            print("average closs:", sum_closs / configs.rounds)
            print("average aloss:", sum_aloss / configs.rounds)
            print("total accuracy:", sum_accuracy)
            print("total cost:", sum_cost)
            print("total round time:", sum_round_time)

    plt.plot(rewards)
    plt.ylabel("Reward")
    plt.xlabel("Episodes")
    # plt.savefig("Rewards.png", dpi=200)
    plt.show()

    # plt.plot(actions)
    # plt.ylabel("action")
    # plt.xlabel("Episodes")
    # # plt.savefig("actions.png", dpi=200)
    # plt.show()

    plt.plot(alosses)
    plt.ylabel("aloss")
    plt.xlabel("Episodes")
    # plt.savefig("Rewards.png", dpi=200)
    plt.show()

    plt.plot(closses)
    plt.ylabel("closs")
    plt.xlabel("Episodes")
    # plt.savefig("Rewards.png", dpi=200)
    plt.show()

    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("Episodes")
    # plt.savefig("Rewards.png", dpi=200)
    plt.show()

    plt.plot(round_times)
    plt.ylabel("round time")
    plt.xlabel("Episodes")
    # plt.savefig("Rewards.png", dpi=200)
    plt.show()

    plt.plot(accuracies)
    plt.ylabel("accuracy")
    plt.xlabel("Episodes")
    # plt.savefig("Rewards.png", dpi=200)
    plt.show()

    # writer1.writerow(rewards)
    # writer1.writerow(actions)
    # writer1.writerow(alosses)
    # writer1.writerow(closses)
    # writer1.writerow(accuracies)
    # writer1.writerow(payments)
    # writer1.writerow(round_times)
    csvFile1.close()
    C_loss.to_csv('DRL_closs.csv', index=None)

def select_participant(total: int, select: int, latest_participant: list, enable_sv: bool, sv: list):
    p = 0.8
    print('[INFO] Select {} in {}'.format(total, select))
    if not enable_sv :
        # random select
        print('[INFO] Random select active')
        return random.sample(range(0, total), select)
    else :
        print('[INFO] SV select active')
        # enable sv select
        # select 80% unit with exploit strategy, and other use explore strategy
        exploit = int(p * select)
        explore = select - exploit
        print('[INFO] Exploit {}'.format(exploit))
        print('[INFO] Explore {}'.format(explore))
        silence = [i for i, x in enumerate(latest_participant) if x == -1]
        print('[INFO] Client not be selected {}'.format(silence))
        explored = [i for i in range(0, total) if i not in silence]
        print('[INFO] Client has been selected {}'.format(explored))
        if len(silence) == 0:
            # nothing left
            print('[INFO] All clients are participants')
            return list(map(sv.index, heapq.nlargest(select, sv)))
        else:
            print('[INFO] Still has client not is a participant')
            max_num_index_list = []
            if len(silence) < explore:
                max_num_index_list = heapq.nlargest(exploit + explore - len(silence), range(len(sv)), sv.__getitem__)
                print('[INFO] Get top sv clients: {}'.format(max_num_index_list))
                return max_num_index_list + silence
            else :
                max_num_index_list = heapq.nlargest(exploit, range(len(sv)), sv.__getitem__)
                print('[INFO] Get top sv clients: {}'.format(max_num_index_list))
                temp1 = set(silence)
                temp2 = set(max_num_index_list)
                diff = temp1.difference(temp2)
                explore_list = list(diff)[:explore]
                return max_num_index_list + explore_list

def Hand_control():
    configs = Configs()
    env = Env(configs)
    sv = [1 for i in range(env.configs.user_num)]
    latest_participant = [-1 for i in range(env.configs.user_num)]
    sv_acc = []
    sv_loss = []
    sv_cost = []
    # alpha * past + beta * current
    alpha = 0.75
    beta = 0.25
    threshold = 12
    for i in range(configs.task_repeat_time):
        print("####### This is the {} repeat task ########".format(i))
        cur_bid = env.reset()
        for t in range(configs.rounds):
            # for each round set the init participant to zero
            local_ep_list = [0 for i in range(env.configs.user_num)]
            selected = select_participant(env.configs.user_num, env.configs.unit, latest_participant, True, sv)
            print('[INFO] Selected usr info: {}'.format(selected))
            # update participant info
            for s in selected:
                latest_participant[s] = i
                local_ep_list[s] = 1
            action = np.array(local_ep_list) / 5
            # print(action)
            selected_user_sv, acc, loss = env.step(action, t)
            sv_acc.append(acc)
            print('[INFO] Accuracy: {}'.format(sv_acc))
            sv_loss.append(loss)
            print('[INFO] Loss: {}'.format(sv_loss))
            cost = 0
            for i in selected:
                if cost < env.configs.exec_speed[i]:
                    cost = env.configs.exec_speed[i]
            sv_cost.append(cost if len(sv_cost) == 0 else sv_cost[-1] + cost)
            print('[INFO] Cost: {}'.format(sv_cost))

            # print("The lenth of all_idx {} is {}:".format(all_idx, len(all_idx)))
            # print("The lenth of selected_idx_list {} is {}:".format(selected_idx_list, len(selected_idx_list)))
            print("The lenth of sv {} is {}:".format(selected_user_sv, len(selected_user_sv)))
            if env.configs.baseline == 1:
                for index in range(len(selected)):
                    sv[selected[index]] = selected_user_sv[index] * alpha + sv[selected[index]] * beta
                print('[INFO] Only SV {}'.format(sv))
            elif env.configs.baseline == 2:
                for index in range(len(selected)):
                    sv[selected[index]] = (selected_user_sv[index] * alpha + sv[selected[index]] * beta) / env.configs.exec_speed[selected[index]] * threshold
                print('[INFO] SV with time concern {}'.format(sv))
            else:
                pass
    fig1 = plt.figure('Fig 1', figsize=(20, 10))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax1.plot(np.arange(1, env.configs.rounds + 1).astype(dtype=np.str), sv_acc, color='red', linestyle='--', marker='x')
    ax1.set_title('SV Accuracy')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')
    ax2 = fig1.add_subplot(1, 2, 2)
    ax2.plot(np.arange(1, env.configs.rounds + 1).astype(dtype=np.str), sv_cost, color='red', linestyle='--', marker='o')
    ax2.set_title('SV Time Cost')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Time Cost')

    plt.show()

    df_cost = pd.DataFrame(sv_cost)
    df_cost.to_csv('v1_sv_cost.csv')
    df_accuracy = pd.DataFrame(sv_acc)
    df_accuracy.to_csv('v1_sv_accuracy.csv')
    df_loss = pd.DataFrame(sv_loss)
    df_loss.to_csv('v1_sv_loss.csv')
    # TODO waiting for data storage
    # recording = recording.append([{'state history': state_list, 'action history': action_list, 'reward history':reward_list, 'acc increase hisotry': performance_increase_list, 'time hisotry': time_list, 'energy history': energy_list, 'social welfare': np.sum(reward_list), 'accuracy': np.sum(performance_increase_list), 'time': np.sum(time_list), 'energy': np.sum(energy_list)}])
    # recording.to_csv('Hand_control_result.csv')

def greedy():
    configs = Configs()
    env = Env(configs)
    csvFile1 = open("recording-Greedy_" + "Client_" + str(configs.user_num) + ".csv", 'w', newline='')
    writer1 = csv.writer(csvFile1)

    accuracies = []
    payments = []
    round_times = []
    rewards = []
    Actionset_list = []

    for EP in range(configs.EP_MAX):
        print(EP)
        print('---------------------')
        cur_bid = env.reset_for_greedy()
        cur_state = np.append(cur_bid, 0)

        recording = []
        # recording.append(cur_state)

        sum_accuracy = 0
        sum_cost = 0
        sum_round_time = 0
        sum_reward = 0
        sum_energy = 0

        if len(Actionset_list) < 20:    # action in first 20 episode is randomly chose
            actionset = np.random.random(configs.rounds * configs.A_DIM)
            actionset = actionset.reshape(configs.rounds, configs.A_DIM)

            for t in range(configs.rounds):
                action = actionset[t]
                tep = action < 0.2
                while tep.all():
                    action = np.random.random(5)
                    tep = action < 0.2
                    actionset[t] = action
                reward, next_bid, delta_accuracy, cost, round_time, int_action, energy = env.step(action)

                sum_accuracy += delta_accuracy
                sum_cost += cost
                sum_round_time += round_time
                sum_reward += reward
                sum_energy += energy

                # recording.append(int_action)
                # recording.append(reward)
                # recording.append(next_state)

                next_state = np.append(next_bid, t + 1)
                cur_state = next_state

            print("------------------------------------------------------------------------")
            print('instant ep:', (EP + 1))


        else:
            tep = np.random.random(1)[0]
            if tep <= 0.2:    # 20% to randomly choose action

                actionset = np.random.random(configs.rounds * configs.A_DIM)
                actionset = actionset.reshape(configs.rounds, configs.A_DIM)

                for t in range(configs.rounds):
                    action = actionset[t]
                    reward, next_bid, delta_accuracy, cost, round_time, int_action, energy = env.step(action)

                    sum_accuracy += delta_accuracy
                    sum_cost += cost
                    sum_round_time += round_time
                    sum_reward += reward
                    sum_energy += energy
                    next_state = np.append(next_bid, t + 1)
                    cur_state = next_state

                print("------------------------------------------------------------------------")
                print('instant ep:', (EP + 1))

            else:     # 80% to choose the Max-R action (Greedy)
                actionset = Actionset_list[0][0]
                sum_reward = Actionset_list[0][1]
                sum_accuracy = Actionset_list[0][2]
                sum_cost = Actionset_list[0][3]
                sum_energy = Actionset_list[0][4]

        recording.append(sum_reward * 10)
        recording.append(sum_accuracy)
        recording.append(sum_cost)
        recording.append(sum_energy)
        writer1.writerow(recording)

        # if action-set is unchanged (80% greedy), then remove it and re-add it with its new reward in this round
        # if action-set is changed (20% random), then add it to the actionset-list
        for one in Actionset_list:
            if (one[0] == actionset).all():
                Actionset_list.remove(one)

        # add the actionset in this round and sort actionset-list by Reward in descending order
        Actionset_list.append((actionset, sum_reward, sum_accuracy, sum_cost, sum_energy))
        Actionset_list = sorted(Actionset_list, key=lambda x: x[1], reverse=True)
        print("ActionSet-List:", Actionset_list)

        # if action-set is unchanged (80% greedy),the actionset-list = 20 and no one will pop
        # if action-set is changed (20% random), pop the last actionset (sorted with Min-R)
        if len(Actionset_list) > 20:
            Actionset_list.pop()
    csvFile1.close()


if __name__ == '__main__':
    # import datetime
    # start = datetime.datetime.now()
    # DRL_train()
    # end = datetime.datetime.now()
    # print(end-start)

    # fed_avg()
    # DRL_inference('Nonecifar_acc2021-01-11')
    # Greedy_myopia()

    seed = 0
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

    Hand_control()
    # greedy()
