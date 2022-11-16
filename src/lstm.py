import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim

def transform_data(
        stocks: pd.DataFrame,
        fields: list[str],
        seq_len: int
    ):
    pass