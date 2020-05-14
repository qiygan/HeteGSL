server, gpu_id = 'Xy', 0
server, gpu_id = 'S5', 3
import os
import sys

sys.path.append('../../')
cur_path = os.getcwd()

from utils.util_funcs import *

python_command = shell_init(server=server, gpu_id=gpu_id)
from utils import Results_dealer

print(os.getcwd())
from models.IDGL import train_idgl

import time
import pickle
import subprocess


class IDGL_Config:
    # model configs
    num_head = 4
    num_hidden = 8
    epochs = 300
    # epochs = 10
    seed = 2020
    weight_decay = 5e-4
    ori_ratio = 0.8
    lr = 0.01
    # other settings
    gpu = 0  # -1 to use cpu
    out_path = '/home/zja/PyProject/HeteGSL/results/IDGL/'
    exp_name = 'IDGL_res_'

    def __init__(self, dataset='cora'):
        self.dataset = dataset


def grid_search():
    return


def grid_tune_single_var(to_be_tuned, para_ind, run_times, resd):
    def _generate_lr_list(para_ind='all'):
        para_set = {}
        para_set['all'] = [0.005]
        return para_set[para_ind]

    # to be tuned parameters
    if to_be_tuned == 'lr':
        tuning_set = _generate_lr_list()

    start_time = time.strftime('%m-%d %H-%M-%S', time.localtime())
    # Start tuning
    for para_i in tqdm_fixed(range(len(tuning_set)), desc=to_be_tuned):
        # for para_i in tqdm_fixed(range(len(tuning_set)).__reversed__(), desc=to_be_tuned):  # Reversed
        para = tuning_set[para_i]
        para_string = '{:.3f}'.format(para) if isinstance(para, float) else para
        settings = '{}={}'.format(to_be_tuned, para_string)
        print('\nRuning :'.format(settings))
        for i in range(run_times):
            # * ================ Default configs ================
            # Model config
            args = IDGL_Config(dataset)
            # * ================= Modify config =================
            args.seed = i
            args.exp_name = args.exp_name + start_time + '.txt'
            exec("args.{} = tuning_set[para_i]".format(to_be_tuned))
            # * ================ Start Running ===================
            print(' <seed={}>'.format(args.seed), end='')
            command_line = gen_run_commands(python_command, cur_path + '/train.py', args)
            print(command_line)
            result = subprocess.run(command_line, stdout=subprocess.PIPE, shell=True)
            # print(result.stdout)
            # * ================ Result Processing ===============
            # fname = '{}.dat'.format(paths['out_path'])
            # with open(fname, 'rb') as handle:
            #     data_loaded = pickle.load(handle)
            #     res_dict, epoch_res = data_loaded['res_dict'], data_loaded['epoch_dict']
            # * =================================================
        resd.calc_mean_std(args.out_path + args.exp_name)
    return pic_path


# * ============ HyperParaTuning Variables ==========
to_be_tuned = 'lr'
para_ind = 'none'
dataset = 'cora'
run_times = 3
# * ================ Model Variables ================
# File paths
log_path = '../results/IDGL'

# * ============== Initialization ===================
resd = Results_dealer(dataset, '../results/')

# * ============== HyperParaTuning ===================
start_time = time.time()
pic_path = grid_tune_single_var(to_be_tuned, para_ind, run_times, resd)
tuning_time = time.time() - start_time
print('Hyper-paramter tuning finished!! tuning time ={}\nPic_path = {} ,'
      .format(time2str(tuning_time), pic_path))
# Save results to excel file
# fname = dataset + mode_name + '_mean_results.txt'
# resd.result_to_exl(fname)
# * =============== Server Commands ===================
# python /home/zja/PyProject/RLBasedGSL
