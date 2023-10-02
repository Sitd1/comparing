import numpy as np
import pandas as pd
from tqdm import tqdm


def tqdm_(iterable, do_tqdm=True):
    if do_tqdm:
        return tqdm(iterable)
    return iterable


def get_bootstraped_data(
        control_data: pd.Series,
        n: int = 1000,
        statistic='median',
        do_tqdm=False
):
    """
        statistic: 'median', 'mean'
    """
    return np.array([control_data.sample(frac=1, replace=True).__getattribute__(statistic)()
                     for _ in tqdm_(range(n), do_tqdm=do_tqdm)])
