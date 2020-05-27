import numpy as np


def shell_init(server='S5', gpu_id=0):
    '''
    Features
    Ignore wanrnings
    Ignore numpy warnings
    Set paths
    :param gpu_id:
    :return:
    '''
    import os
    import warnings
    np.seterr(invalid='ignore')
    warnings.filterwarnings("ignore")
    if server == 'Xy':
        python_command = '/home/chopin/zja/anaconda/bin/python'
    elif server=='Colab':
        python_command = 'python'
    else:
        python_command = '~/anaconda3/bin/python'
        if gpu_id > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    return python_command


def seed_init(seed):
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
    def _write_dict(d, f):
        for key in d.keys():
            if isinstance(d[key], dict):
                f.write(str(d[key]) + '\n')

    with open(f_path, 'a+') as f:
        f.write('\n')
        _write_dict(d, f)


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


def exists_zero_lines(h):
    import torch
    zero_lines = torch.where(torch.sum(h, 1) == 0)[0]
    if len(zero_lines) > 0:
        # raise ValueError('{} zero lines in {}s!\nZero lines:{}'.format(len(zero_lines), 'emb', zero_lines))
        print(f'{len(zero_lines)} zero lines !\nZero lines:{zero_lines}')
        return True
    return False
