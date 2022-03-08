import numpy as np
import random

def bs_corrs(corr, Nbs, Mbs=None, seed=None, return_bs_list=False, return_mbs=False):
    ''' generate bootstrap resampling of correlation function data
        Args:
            - corr: numpy array of data (Ncfg, Nt, ...)
            - Nbs:  the number of bootstrap samples to generate
            - Mbs:  the number of random draws per bootstrap to generate
                    if Mbs != Ncfg, you will have to appropriately rescale
                    the fluctuations by sqrt( Mbs / Ncfg)
            - seed: a string that will be hashed to seed the random number generator

        Return:
            return_mbs=False
                corr_bs: an array of shape (Nbs, Nt, ...)
            return_mbs=True
                corr_bs: an array of shape (Nbs, Mbs, Nt, ...)
            return_bs_list=True
                corr_bs, bs_list.shape = (Nbs, Mbs)
    '''

    Ncfg = corr.shape[0]
    if Mbs:
        m_bs = Mbs
    else:
        m_bs = Ncfg

    # seed the random number generator
    if seed:
        ''' ChrisK - here is where we should use your hashing '''
        random.seed(seed)
        seed_int = random.randint(1,1e6)
        np.random.seed(seed_int)

    # make BS list
    bs_list = np.random.randint(Ncfg, size=[Nbs, m_bs])

    # make BS corrs
    corr_bs = np.zeros(tuple([Nbs, m_bs]) + corr.shape[1:],dtype=corr.dtype)
    for bs in range(Nbs):
        corr_bs[bs] = corr[bs_list[bs]]

    if return_bs_list:
        return corr_bs, bs_list
    else:
        return corr_bs
