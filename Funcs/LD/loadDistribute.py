import pandas as pd
import numpy as np

def loadDistribute(PF_data_pin, sec_interval, params):
    load_avg = PF_data_pin.reset_index(drop=True)
    # load_avg.columns = [f'N{i}' for i in range(1, params.nums_Tss+1)]
    load_avg.reset_index(drop=True, inplace=True)
    load_avg['groupBy'] = load_avg.index.values.astype('int')//sec_interval
    load_avg = load_avg.groupby('groupBy').mean()
    load_avg.reset_index(drop=True, inplace=True)
    # load_avg.drop(columns=['groupBy'], inplace=True)
    return load_avg