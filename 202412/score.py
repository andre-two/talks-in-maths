import pandas as pd
import geopandas as gpd

import numpy as np
from scipy.stats import norm


def get_z_alpha_2( alpha_2):
    return norm.ppf(1-alpha_2)

def lambda_min( alpha_2, k):
    lambda_min = (get_z_alpha_2(alpha_2)**2) / (k**2)
    return lambda_min


def credibility(n_sin, alpha_2 = .05, k = .1):
    if np.isnan(n_sin):
        return 0
    else:
        z = min(np.sqrt(n_sin / lambda_min(alpha_2, k)),1.)
        return z

