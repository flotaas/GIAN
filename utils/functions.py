import numpy as np
import torch
import torch.nn.functional as F
import random
import pynvml
import logging
global logger
logger = logging.getLogger('MMSA')


def list_to_str(src_list):
    dst_str = ""
    for item in src_list:
        dst_str += " %.4f " %(item) 
    return dst_str

def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str

def calculate_AUILC(metrics_list):
    result = 0
    for i in range(len(metrics_list)-1):
        result += (metrics_list[i] + metrics_list[i+1]) * 0.1 / 2
    return result

def rbf_kernel(x, y, gamma=1.0):
    """
    Radial Basis Function (RBF) kernel.
    Args:
        x, y: Tensors of shape (B, T, D)
        gamma: Kernel coefficient
    Returns:
        Kernel matrix of shape (B, T, T)
    """
    dist = torch.cdist(x, y, p=2) ** 2  # Pairwise squared distances
    return torch.exp(-gamma * dist)

def mmd_loss(x, y, gamma=0.5):
    """
    Maximum Mean Discrepancy (MMD) loss between two distributions.
    Args:
        x, y: Tensors of shape (B, T, D)
        gamma: Kernel coefficient
    Returns:
        MMD loss (scalar)
    """
    K_xx = rbf_kernel(x, x, gamma)  # Kernel within x
    K_yy = rbf_kernel(y, y, gamma)  # Kernel within y
    K_xy = rbf_kernel(x, y, gamma)  # Kernel between x and y
    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()

def kl_divergence(p, q):
    """
    KL divergence between two probability distributions.
    Args:
        p, q: Tensors of shape (B, T, D), representing probabilities
    Returns:
        KL divergence (scalar)
    """
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    return torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8))) / p.size(0)

def js_divergence(p, q):
    """
    Jensen-Shannon divergence between two distributions.
    Args:
        p, q: Tensors of shape (B, T, D), representing probabilities
    Returns:
        JS divergence (scalar)
    """
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    m = 0.5 * (p + q)
    return 0.5 * (torch.sum(p * torch.log((p + 1e-8) / (m + 1e-8))) +
                  torch.sum(q * torch.log((q + 1e-8) / (m + 1e-8)))) / p.size(0)

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def assign_gpu(gpu_ids, memory_limit=1e16):
    if len(gpu_ids) == 0 and torch.cuda.is_available():
        # find most free gpu
        pynvml.nvmlInit()
        n_gpus = pynvml.nvmlDeviceGetCount()
        dst_gpu_id, min_mem_used = 0, memory_limit
        for g_id in range(n_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        logger.info(f'Found gpu {dst_gpu_id}, used memory {min_mem_used}.')
        gpu_ids.append(dst_gpu_id)
    # device
    using_cuda = len(gpu_ids) > 0 and torch.cuda.is_available()
    # logger.info("Let's use %d GPUs!" % len(gpu_ids))
    device = torch.device('cuda:%d' % int(gpu_ids[0]) if using_cuda else 'cpu')
    return device

def count_parameters(model):
    answer = 0
    for p in model.parameters():
        if p.requires_grad:
            answer += p.numel()
            # print(p)
    return answer

class Storage(dict):
    """
    A Storage object is like a dictionary except `obj.foo` can be used inadition to `obj['foo']`
    ref: https://blog.csdn.net/a200822146085/article/details/88430450
    """
    def __getattr__(self, key):
        try:
            return self[key] if key in self else False
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __str__(self):
        return "<" + self.__class__.__name__ + dict.__repr__(self) + ">"