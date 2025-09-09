#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   stopping_volt_fillna.py
@Time    :   2025/02/18 22:15:44
@Author  :   
'''

# here put the import lib

import matplotlib.pyplot as plt
import re
import pandas as pd

def stopping_volt_fillna(vs,DFs):
    for col in vs.columns[11:].values:
            match = re.match('Volt_N(\d+)',col)
            num_train = int(match.group(1))
            # print(num_train)
            for idx in vs[col].index.values:
                # print(idx)
                if pd.isna(vs.loc[idx,col]):
                    if vs.index.values[0] == 'H0':
                        match = re.match('H(\d+)',idx)
                        num_horizon = int(match.group(1))
                    elif vs.index.values[0] == 0:
                        num_horizon = idx
                    current_DF = DFs[num_horizon]
                    # print(num_train in current_DF['name'].values.astype('int'))
                    # print(list(current_DF['name'].values.astype('int')))
                    if (num_train in list(current_DF['name'].values.astype('int'))):

                        stoppingTss = current_DF.loc[current_DF['name']==num_train,'stopping'].values[0]
                        if stoppingTss > 0 :
                            # print(stoppingTss)
                            vs.loc[idx,col] = vs.loc[idx,'Volt_N'+str(stoppingTss)]
                else: pass
    return vs