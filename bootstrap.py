import numpy as np
from numpy.random import Generator, SeedSequence, PCG64
import hashlib


def get_rng(seed: str):
    """Generate a random number generator based on a seed string."""
    # Over python iteration the traditional hash was changed. So, here we fix it to md5
    hash = hashlib.md5(seed.encode("utf-8")).hexdigest()  # Convert string to a hash
    seed_int = int(hash, 16) % (10 ** 6)  # Convert hash to an fixed size integer
    print("Seed to md5 hash:", seed, "->", hash, "->", seed_int)
    # Create instance of random number generator explicitly to ensure long time support
    # PCG64 -> https://www.pcg-random.org/
    # see https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
    rng = Generator(PCG64(SeedSequence(seed_int)))
    return rng


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
    rng = get_rng(seed) if seed else np.random.default_rng()

    # make BS list: [low, high)
    bs_list = rng.integers(low=0, high=Ncfg, size=[Nbs, m_bs])

    # make BS corrs
    corr_bs = np.zeros(tuple([Nbs, m_bs]) + corr.shape[1:], dtype=corr.dtype)
    for bs in range(Nbs):
        corr_bs[bs] = corr[bs_list[bs]]

    # if return_mbs, return (Nbs, Mbs, Nt, ...) array
    # otherwise, return mean over Mbs axis
    if return_mbs:
        bs_mean   = corr_bs.mean(axis=(0,1))
        d_corr_bs = corr_bs - bs_mean
        corr_bs   = bs_mean + d_corr_bs * np.sqrt( m_bs / Ncfg)
    else:
        corr_bs   = corr_bs.mean(axis=1)
        bs_mean   = corr_bs.mean(axis=0)
        d_corr_bs = corr_bs - bs_mean
        corr_bs   = bs_mean + d_corr_bs * np.sqrt( m_bs / Ncfg)

    if return_bs_list:
        return corr_bs, bs_list
    else:
        return corr_bs
