server, gpu = 'Xy', 0
server, gpu = 'S5', 3
server, gpu = 'Ali', 3
server, gpu = 'Ali', -1
server, gpu = 'S3', -2
server, gpu = 'S3', -2
server, gpu = 'S3', 0
server, gpu = 'Colab', 0
import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path.split('src')[0]
sys.path.append(root_path + 'src')
os.chdir(root_path)

from models.IDGL.config import IDGL_Config
from utils.util_funcs import *

python_command = shell_init(server=server, gpu_id=gpu)
print(python_command)
from utils import Results_dealer

print(os.getcwd())
from models.IDGL import train_idgl

import time
import pickle
import subprocess


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
            args = IDGL_Config(dataset, gpu)
            mode_name = 'IDGL'  # 0
            # args.pretrain_epochs, args.exp_name, args.gpu = 1, f'<{mode_name}>_wo_pretrain', 1  # 1
            # args.early_stop, args.exp_name, args.gpu = 0, f'<{mode_name}>wo.EarlyStop', 1  # 2
            # args.alpha, args.exp_name, args.gpu = 0, f'<{mode_name}>wo.Dirichlet', 2  # 3
            # args.alpha, args.exp_name, args.gpu = 0, f'<{mode_name}>wo.sparsity', 3  # 4
            # args.ngrl, args.exp_name, args.gpu = 1, f'<{mode_name}>Normed_graph', 0  # 5
            args.exp_name = f'<{mode_name}>wo.GraphReg'
            # * ================= Modify config =================
            args.seed = i
            args.exp_name = args.exp_name + start_time + '.txt'
            exec("args.{} = tuning_set[para_i]".format(to_be_tuned))
            # * ================ Start Running ===================
            print(' <seed={}>'.format(args.seed), end='')
            command_line = gen_run_commands(python_command, cur_path.replace(' ', '\ ') + '/train.py', args)
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
    return None


# * ============ HyperParaTuning Variables ==========
to_be_tuned = 'lr'
para_ind = 'none'  # @
dataset = 'cora'
dataset = 'citeseer'
run_times = 10
# * ============== Initialization ===================
resd = Results_dealer(dataset, '../results/')

# * ============== HyperParaTuning ===================
start_time = time.time()
grid_tune_single_var(to_be_tuned, para_ind, run_times, resd)
tuning_time = time.time() - start_time
# * =============== Server Commands ===================
# python ~/PyProject/HeteGSL/src/models/IDGL/tuneIDGL.py
