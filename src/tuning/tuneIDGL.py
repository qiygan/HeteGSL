server = 'S4'
server = 'S5'
server = 'Xy'
from utils.util_funcs import shell_init, tqdm_fixed, time2str


class IDGL_Config:
    model = 'IDGL'
    # model configs
    lr = 0.01
    weight_decay = 5e-4
    num_head = 4
    lamda = 0.8
    epochs = 300
    seed = 2020
    # other settings
    out_path = '/results/IDGL/'
    def __init__(self, dataset='cora'):
        self.dataset = dataset


shell_init(server='S5', gpu_id=3)
import numpy as np
import time
from models.IDGL import train_idgl


def grid_search():
    return


def grid_tune_single_var(to_be_tuned, para_ind, run_times, resd):
    def _generate_size_list(para_ind='all'):
        para_set = {}
        para_set['all'] = 1
        para_set[2] = 2
        return para_set[para_ind]

    # to be tuned parameters
    if to_be_tuned == 'size':
        tuning_set = _generate_size_list()

    start_time = time.strftime('%m-%d %H:%M%S', time.localtime())
    # Start tuning
    for para_i in tqdm_fixed(range(len(tuning_set)), desc=to_be_tuned):
        # for para_i in tqdm_fixed(range(len(tuning_set)).__reversed__(), desc=to_be_tuned):  # Reversed
        para = tuning_set[para_i]
        para_string = '{:.3f}'.format(para) if isinstance(para, float) else para
        settings = '{}={}'.format(to_be_tuned, para_string)
        print('\nRuning :'.format(settings))
        for i in range(run_times):
            np.random.seed(i)
            seed = i
            # * ================ Default configs ================
            # Model config
            args = IDGL_Config(dataset)
            # * ================= Modify config =================
            exec("%s = tuning_set[para_i]" % 'args.' + to_be_tuned)
            # * ================ Start Running ===================
            print('\nRun{} {}'.format(i, exp_name), end='')
            print(' <seed={}>'.format(seed), end='')
            command_line = [python_command, 'trainNENS.py', '--config', con_fname]
            result = subprocess.run(command_line, stdout=subprocess.PIPE)
            # print(result.stdout)
            # * ================ Result Processing ===============
            fname = '{}.dat'.format(paths['out_path'])
            with open(fname, 'rb') as handle:
                data_loaded = pickle.load(handle)
                res_dict, epoch_res = data_loaded['res_dict'], data_loaded['epoch_dict']
            # * =================================================
        draw_input = {'mode_name': mode_name, 'run_times': run_times,
                      'start_time': start_time, 'dataset': dataset,
                      'exp_name': exp_name, 'settings': settings,
                      'config_dict': config_dict}
        # resd.calc_and_write_mean_results(mode_name, res_list)
        pic_path = resd.process_epoch_data(epoch_list, draw_input)
    return pic_path


# * ============ HyperParaTuning Variables ==========
to_be_tuned = 'alpha'
para_ind = 'none'
dataset = 'dblp'
dataset = 'acm'
# * ================ Model Variables ================
# File paths
log_path = '../results/IDGL'

# * ============== Initialization ===================
resd = Results_dealer(dataset, '../results/')

# * ============== HyperParaTuning ===================
start_time = time.time()
pic_path = grid_tune_single_var(to_be_tuned, para_ind, resd)
tuning_time = time.time() - start_time
print('Hyper-paramter tuning finished!! tuning time ={}\nPic_path = {} ,'
      .format(time2str(tuning_time), pic_path))
# Save results to excel file
# fname = dataset + mode_name + '_mean_results.txt'
# resd.result_to_exl(fname)
# * =============== Server Commands ===================
# python /home/zja/PyProject/RLBasedGSL
