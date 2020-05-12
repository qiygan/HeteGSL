def shell_init(server='S5', gpu_id=0, f_prefix=''):
    '''
    Ignore wanrnings
    Set paths
    :param gpu_id:
    :return:
    '''
    import os
    import warnings

    warnings.filterwarnings("ignore")
    if server == 'Xy':
        SOURCE_PATH = '/home/chopin/zja/PyProject/HeteGSL/' + f_prefix
        # SOURCE_PATH = '/home/zja/PyProject/HeteRLWalk/src_jhy/'
    else:
        SOURCE_PATH = '/home/zja/PyProject/HeteGSL/' + f_prefix
    os.chdir(SOURCE_PATH)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


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


def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)
