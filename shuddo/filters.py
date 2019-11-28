# Copyright (c) 2019, Md Imam Hossain (emamhd at gmail dot com)
# see LICENSE.txt for details

"""
Data filtering functions
"""

def S_moving_average_data(_data_list, _smoothing=1):
    """
    Returns moving average data without data lag, use the smoothing factor to get required overall smoothing
    """    

    ma_data = []    
    ds = len(_data_list)
    s = _smoothing
    mas = int((len(_data_list)*0.02) * s)
    fc = int(mas/2)
    fmas = fc * 2

    for i in range(ds):
        if i < fc:
            db = _data_list[:i+1]
            nfc = len(db)
            ma_data.append(sum(db)/nfc)
        elif i >= fc:
            if i < (ds - fc):
                ma_data.append(sum(_data_list[i-fc:i+fc])/fmas)
            else:
                db = _data_list[i:]
                nfc = len(db)
                ma_data.append(sum(db)/nfc)
    
    return ma_data
