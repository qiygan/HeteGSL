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

from models.HGSL.config import HGSL_Config
from utils.util_funcs import *

python_command = shell_init(server=server, gpu_id=gpu)
print(python_command)
from utils import Results_dealer

print(os.getcwd())

import time
import pickle
import subprocess
import multiprocessing


def grid_search():
    return


def grid_tune_single_var(to_be_tuned, para_ind, run_times, mode_name):
    def _generate_lr_list(para_ind='all'):
        para_set = {}
        para_set['all'] = [0.005]
        return para_set[para_ind]

    resd = Results_dealer(dataset, '../results/')
    # to be tuned parameters
    if to_be_tuned == 'lr':
        tuning_set = _generate_lr_list()

    start_time = time.strftime('%m-%d %H-%M-%S', time.localtime())
    # Start tuning
    for para_i in tqdm_fixed(range(len(tuning_set)), desc=to_be_tuned):
        para = tuning_set[para_i]
        para_string = f'{para:.3f}' if isinstance(para, float) else para
        settings = f'{to_be_tuned}={para_string}'
        print('\nRuning :'.format(settings))
        for i in range(run_times):
            # * ================= Modify config =================
            # Default config
            args = HGSL_Config(dataset, gpu)
            args = modify_config(args, mode_name, start_time, i)
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


def modify_config(args, mode_name, start_time, seed):
    args.seed = seed
    model = 'IDGL'
    if mode_name == 'wo_pretrain':
        args.pretrain_epochs, args.exp_name, args.gpu = 1, f'<{model}>_wo_pretrain', 1  # 1
    elif mode_name == 'wo_EarlyStop':
        args.early_stop, args.exp_name, args.gpu = 0, f'<{model}>wo_EarlyStop', 1  # 2
    elif mode_name == 'wo_Dirichlet':
        args.alpha, args.exp_name, args.gpu = 0, f'<{model}>wo_Dirichlet', 2  # 3
    elif mode_name == 'wo_sparsity':
        args.alpha, args.exp_name, args.gpu = 0, f'<{model}>wo_sparsity', 3  # 4
    elif mode_name == 'normed_graph':
        args.ngrl, args.exp_name, args.gpu = 1, f'<{model}>Normed_graph', 0  # 5
    args.exp_name = args.exp_name + start_time + '.txt'
    return args


# * ============ HyperParaTuning Variables ==========
to_be_tuned = 'lr'
para_ind = 'none'  # @
dataset = 'cora'
dataset = 'citeseer'
run_times = 10
# * ============== HyperParaTuning ===================
start_time = time.time()
exps = ['wo_pretrain', 'wo_EarlyStop', 'wo_Dirichlet', 'wo_sparsity', 'normed_graph']
exps = ['wo_pretrain', 'wo_EarlyStop', 'wo_Dirichlet']
process_list = []
for mode in exps:
    _ = multiprocessing.Process(target=grid_tune_single_var,
                                args=(to_be_tuned, para_ind, run_times, mode))
    process_list.append(_)
    _.start()
for _ in process_list:
    _.join()
print(f'tuning_time={time2str(time.time() - start_time)}')
# * =============== Server Commands ===================
# python ~/PyProject/HGSL/src/models/IDGL/tuneHGSL.py


