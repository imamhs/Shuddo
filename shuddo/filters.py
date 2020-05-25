# Copyright (c) 2019, Md Imam Hossain (emamhd at gmail dot com)
# see LICENSE.txt for details

"""
Data filtering functions
"""

def S_moving_average_data(_data_list, _smoothing=1):
    """
    Returns moving average data without data lag.
    Use the smoothing factor to get required overall smoothing
    """    

    ma_data = []    
    ds = len(_data_list)
    s = _smoothing
    mas = int((ds * 0.02) * s)
    fc = int(mas/2)
    fmas = fc * 2

    for i in range(ds):
        if i < fc:
            db = _data_list[:i+i+1]
            nfc = len(db)
            ma_data.append(sum(db)/nfc)
        elif i >= fc:
            if i < (ds - fc):
                ma_data.append(sum(_data_list[i-fc:i+fc+1])/(fmas+1))
            else:
                db = _data_list[i-(ds-i-1):]
                nfc = len(db)
                ma_data.append(sum(db)/nfc)
    
    return ma_data

def S_downsample(_data_list, _factor=1):
    """
    Returns a two dimensional data set with a reduced number of samples.
    Use the sample skipping factor to get required result, the factor tells how many samples to skip for one data sample
    """    
    
    ds_data = []   
    ds = len(_data_list)
    skip_count = 0
    
    for i in range(ds):
        
        if skip_count < _factor:
            skip_count += 1
        else:
            ds_data.append((_data_list[i][0], _data_list[i][1]))
            skip_count = 0
        
    return ds_data
