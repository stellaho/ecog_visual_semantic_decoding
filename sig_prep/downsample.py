from math import gcd

from scipy.signal import resample_poly

def downsample_signal(sig, trg_fs, src_fs):
    factor = gcd(trg_fs, src_fs)
    return resample_poly(sig, trg_fs // factor, src_fs // factor, axis=-1)
