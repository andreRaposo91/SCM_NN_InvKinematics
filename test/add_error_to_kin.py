import random as rnd
import math
import numpy as np

def add_error_to_kin(
    lens_curr,
    lens_prev,
    rnd_scale=0.1,
    curv_scale=0,
    len_scale=0,
    dire_scale=0,
    hist_scale=0,
    ):

    at_rest = 65
    w = 18

    curv_curr = math.sqrt(lens_curr[0]**2 + lens_curr[1]**2 + lens_curr[2]**2 -
        (lens_curr[0]*lens_curr[1] + lens_curr[0]*lens_curr[2] + lens_curr[1]*lens_curr[2])) / 3 / w
    curv_prev = math.sqrt(lens_prev[0]**2 + lens_prev[1]**2 + lens_prev[2]**2 -
        (lens_prev[0]*lens_prev[1] + lens_prev[0]*lens_prev[2] + lens_prev[1]*lens_prev[2])) / 3 / w
    curv_diff = curv_curr - curv_prev

    avg_curr = sum(lens_curr) / 3
    avg_prev = sum(lens_prev) / 3
    lens_diff = avg_curr - avg_prev

    # dire = [c - p for c, p in zip(lens_curr, lens_prev)] # could be used

    cable_err = curv_curr * curv_scale + avg_curr * len_scale + abs(lens_diff) * dire_scale
    pulley_err = lens_diff * dire_scale + (avg_prev - at_rest) * len_scale
    hist_err = curv_diff * np.sign(lens_diff) * hist_scale
    rnd_err = 2*(rnd.random() - 0.5) * rnd_scale

    return np.array([lens*(1 + rnd_err + cable_err + pulley_err + hist_err) for lens in lens_curr])
