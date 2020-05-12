import os
import warnings
import torch


def shell_init(server='S5', gpu_id=0):
    '''
    Ignore wanrnings
    Set paths
    :param gpu_id:
    :return:
    '''
    warnings.filterwarnings("ignore")
    if server == 'Xy':
        SOURCE_PATH = '/home/chopin/zja/PyProject/HeteRLWalk/src/'
        # SOURCE_PATH = '/home/zja/PyProject/HeteRLWalk/src_jhy/'
    else:
        SOURCE_PATH = '/home/zja/PyProject/HeteRLWalk/src/'
    os.chdir(SOURCE_PATH)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


def cos_sim(a, b, eps=1e-8):
    """
    TODO: Self Similarity
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
