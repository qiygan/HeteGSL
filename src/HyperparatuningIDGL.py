server = 'S4'
server = 'S5'
server = 'Xy'
from utils.util_funcs import shell_init

shell_init(server='S5', gpu_id=3)
import numpy as np
import time


def grid_tune(to_be_tuned, para_ind, resd):
    # to be tuned parameters
    # if to_be_tuned == 'size':
    #     tuning_set = _generate_size_list(para_ind)

    start_time = time.strftime('%m-%d %H:%M%S', time.localtime())
    # Start tuning
    for para_i in tqdm_fixed(range(len(tuning_set)), desc=to_be_tuned):
        # for para_i in tqdm_fixed(range(len(tuning_set)).__reversed__(), desc=to_be_tuned):  # Reversed
        para = tuning_set[para_i]
        para_string = '{:.3f}'.format(para) if isinstance(para, float) else para
        settings = '{}={}'.format(to_be_tuned, para_string)
        print('\nRuning :'.format(settings))
        res_list = []
        epoch_list = []
        for i in range(run_times):
            np.random.seed(i)
            seed = i
            # * ================ Default configs ================
            # Optimizer config
            optm_conf = {'opt_method': hp.opt_method, 'alpha': hp.alpha,
                         'lr_decay': hp.lr_decay, 'weight_decay': hp.weight_decay}
            # paths

            exp_name = 'NENS<{}>{}_{:.3f}-[{}]-{}'.format(
                mode_name, dataset, optm_conf['alpha'], settings, time.strftime('%m-%d %H:%M:%S', time.localtime()))
            temp_exp_path = '../tmp/{}'.format(exp_name)
            paths = {'log_path': log_path, 'out_path': temp_exp_path}
            # Train config
            train_conf = \
                {'dataset': dataset, 'exp_name': exp_name,
                 'server': server, 'vis_flag': hp.vis_flag,
                 'eval_freq': hp.eval_freq, 'eval_flag': hp.eval_flag,
                 'train_on_gpu': hp.train_on_gpu,
                 'train_times': train_times, 'seed': seed,
                 }
            # Model config
            model_conf = \
                {'init_dw_emb': hp.init_dw_emb,
                 'cla_method': hp.cla_method, 'cla_layers': hp.cla_layers,
                 'ns_emb_mode': hp.ns_emb_mode, 'conv_method': hp.conv_method,
                 'norm_emb_flag': hp.norm_emb_flag,
                 'dw_feat_only': hp.dw_feat_only}
            # Model paras
            model_paras = \
                {'beta': hp.beta, 'ns_neg_rate': hp.ns_neg_rate,
                 'e_neg_rate': hp.e_neg_rate, 'size': hp.size}
            # * ================= Combine config =================
            config_dict = \
                {'paths': paths, 'train_conf': train_conf,
                 'model_conf': model_conf, 'model_paras': model_paras,
                 'optm_conf': optm_conf}
            # * ================= Modify config =================
            for conf_name in config_dict:
                if to_be_tuned in config_dict[conf_name]:
                    config_dict[conf_name][to_be_tuned] = tuning_set[para_i]
            # * ================= Save config ====================
            # Save exp config
            con_fname = '{}_conf.txt'.format(temp_exp_path)
            save_dict_json(config_dict, con_fname)
            # * ================ Start Running ===================
            print('\nRun{} {}'.format(i, exp_name), end='')
            print(' <seed={}>'.format(seed), end=' ')
            command_line = [python_command, 'trainNENS.py', '--config', con_fname]
            result = subprocess.run(command_line, stdout=subprocess.PIPE)
            # print(result.stdout)
            # * ================ Result Processing ===============
            fname = '{}.dat'.format(paths['out_path'])
            with open(fname, 'rb') as handle:
                data_loaded = pickle.load(handle)
                res_dict, epoch_res = data_loaded['res_dict'], data_loaded['epoch_dict']
            res_list.append(res_dict)
            epoch_df = pd.DataFrame.from_records(epoch_res)
            epoch_list.append(epoch_df)
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
log_path = '../results'
# Train config
from utils.util_funcs import shell_init
shell_init(server='S5', gpu_id=3)
# * ============== Initialization ===================
mode_name = '<IMDB44>Alpha={}'.format(hp.alpha)
# mode_name = 'IMDB_largeEpoch,Alpha={}_cla_layer={}_[neg_edge,ns={},{}]'.format(alpha, cla_layers, e_neg_rate, ns_neg_rate)
print(
    '\ndataset={}\tconv_method={}\tcla_method,layers={}_{}\tneg_edge,ns={},{}\nrun_config={}\trun_times = {}\ttrain_times={} \n'
        .format(dataset, hp.conv_method, hp.cla_method, hp.cla_layers, hp.e_neg_rate, hp.ns_neg_rate, run_config,
                run_times,
                train_times))
resd = Results_dealer(dataset, '../results/')

# * ============== HyperParaTuning ===================
start_time = time.time()
pic_path = grid_tune(to_be_tuned, para_ind, resd)
tuning_time = time.time() - start_time
print('Hyper-paramter tuning finished!! tuning time ={}\nPic_path = {} ,'
      .format(time2str(tuning_time), pic_path))
# Save results to excel file
# fname = dataset + mode_name + '_mean_results.txt'
# resd.result_to_exl(fname)
# * =============== Server Commands ===================
# python /home/zja/PyProject/RLBasedGSL