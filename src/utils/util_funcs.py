def shell_init(server='S5', gpu_id=0, f_prefix=''):
    '''
    Ignore wanrnings
    Set paths
    :param gpu_id:
    :return:
    '''
    import os
    import warnings
    import sys

    warnings.filterwarnings("ignore")
    if server == 'Xy':
        SOURCE_PATH = '/home/chopin/zja/PyProject/HeteGSL/' + f_prefix
        python_command = '/home/chopin/zja/anaconda/bin/python'
    else:
        SOURCE_PATH = '/home/zja/PyProject/HeteGSL/' + f_prefix
        python_command = 'python'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.chdir(SOURCE_PATH)
    sys.path.append(SOURCE_PATH + 'src/')
    return python_command


def seed_init(seed):
    import numpy as np
    import torch
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cos_sim(a, b, eps=1e-8):
    """
    TODO: Self Similarity
    """
    import torch
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def tqdm_fixed(*args, **kwargs):
    from tqdm import tqdm as tqdm_base
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)


def print_dict(d, end_string='\n\n'):
    for key in d.keys():
        if isinstance(d[key], dict):
            print('\n', end='')
            print_dict(d[key], end_string='')
        elif isinstance(d[key], int):
            print('{}: {:04d}'.format(key, d[key]), end=', ')
        elif isinstance(d[key], float):
            print('{}: {:.4f}'.format(key, d[key]), end=', ')
        else:
            print('{}: {}'.format(key, d[key]), end=', ')
    print(end_string, end='')


def write_dict(d, f_path):
    with open(f_path, 'a') as f:
        for key in d.keys():
            if isinstance(d[key], dict):
                f.write('\n', end='')
                print_dict(d[key], end_string='')
            elif isinstance(d[key], int):
                f.write('{}: {:04d}'.format(key, d[key]), end=', ')
            elif isinstance(d[key], float):
                f.write('{}: {:.4f}'.format(key, d[key]), end=', ')
            else:
                f.write('{}: {}'.format(key, d[key]), end=', ')
        f.write(str(res_dict['Env_name']) + '\n')
        f.write(str(res_dict['Train States:']) + '\n')
        f.write(str(res_dict['para_dict']) + '\n')
        f.write('Paper Results:\n' + str(res_dict['Paper M-Results:']) + '\n')
        f.write(str(res_dict['Paper F-Results:']) + '\n')
        f.write('Author Results:\n' + str(res_dict['Author M-Results:']) + '\n')
        f.write(str(res_dict['Author F-Results:']) + '\n\n')
        f.close()


def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def gen_run_commands(python_command='python', prog_path='', conf=None, return_str=True):
    var_list = conf.__dict__.keys()
    if return_str:
        res = python_command + ' ' + prog_path + ' '
        for var in var_list:
            if var[0] != '_':
                val = conf.__dict__[var]
                val_s = "'{}'".format(val) if isinstance(val, str) else str(val)
                res += '--' + var + '=' + val_s + ' '
        return res
    else:
        command_list = [python_command, prog_path]
        for var in var_list:
            if var[0] != '_':
                val = conf.__dict__[var]
                # val_s = "'{}'".format(val) if isinstance(val, str) else str(val)
                # command_list += '--' + var + '=' + val_s + ' '
                command_list.append('--' + var)
                command_list.append(str(val))
    return command_list
