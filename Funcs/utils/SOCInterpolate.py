#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   SOCInterpolate.py
@Time    :   2025/03/30 14:57:47
@Author  :   
'''

# here put the import lib
import numpy as np
import pandas as pd

def SOCInterpolate(dayAheadSOC, sec, offlineHorizon, numTss):
    dayAheadSOC.reset_index(inplace=True, drop = True)
    # dayAheadSOC.index = (range(1, 1+len(dayAheadSOC)))
    # dayAheadSOC.loc[0,:] = [0.5]*numTss
    # dayAheadSOC.sort_index(inplace=True)

    SOCInterpolated = dayAheadSOC.copy()
    SOCInterpolated = SOCInterpolated.astype('float')
    SOCInterpolated.index = np.array(SOCInterpolated.index)*sec
    p = pd.DataFrame(columns=['insertBy'], index= list(range(0, offlineHorizon+1)))
    SOCInterpolated = SOCInterpolated.join(p, how='right')
    # print(SOCInterpolated)
    SOCInterpolated.drop(columns=['insertBy'], inplace=True)
    # SOCInterpolated.interpolate(method='polynomial', order = 2, inplace=True, axis= 0)
    SOCInterpolated.interpolate(method='linear', inplace=True, axis= 0)
    # SOCInterpolated = SOCInterpolated.iloc[1:,:]
    # SOCInterpolated.reset_index(drop = True, inplace = True)
    SOCInterpolated.columns = SOCInterpolated.columns.map(lambda x : int(x.split('_N')[1]))
    return SOCInterpolated