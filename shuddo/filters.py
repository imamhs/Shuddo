# Copyright (c) 2020, Md Imam Hossain (emamhd at gmail dot com)
# see LICENSE.txt for details

"""
Data filtering functions
"""

from math import pi, cos, sin

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

def S_linear_function(_point1, _point2, _npoints):
    """
    Returns a list of points linearly interpolated between point1 and point2 with npoints number of points.
    """

    if _npoints < 1:
        return []

    points = []

    step = 1 / (_npoints-1)
    t = 0
    d = _point2[0] - _point1[0]

    while t <= 1:
        x = _point1[0] + (d*t)
        points.append((x,(_point1[1]*(1-t))+(_point2[1]*t)))
        t += step

    return points

def S_cosine_function(_point1, _point2, _npoints):
    """
    Returns a list of points Cosine interpolated between point1 and point2 with npoints number of points.
    """

    if _npoints < 1:
        return []

    points = []

    step = 1 / (_npoints-1)
    t = 0
    d = _point2[0] - _point1[0]

    while t <= 1:
        x = _point1[0] + (d*t)
        t1 = (1-cos(t*pi))/2
        points.append((x,(_point1[1]*(1-t1))+(_point2[1]*t1)))
        t += step

    return points

def S_filter_data(_data_list, _max, _min):
    """
    returns a filtered data where values are discarded according to max, min limits
    """

    f_data = []
    ds = len(_data_list)

    i = 0

    while i < ds:

        if _data_list[i] >= _max:
            f_data.append(_max)
        elif _data_list[i] <= _min:
            f_data.append(_min)
        else:
            f_data.append(_data_list[i])

        i += 1

    return f_data
