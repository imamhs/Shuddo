# Copyright (c) 2020, Md Imam Hossain (emamhd at gmail dot com)
# see LICENSE.txt for details

"""
Data filtering functions
"""

from math import pi, cos
from shuddo import mining

def S_moving_average_data(_data_list, _smoothing=1):
    """
    Returns moving average data without data lag.
    Use the smoothing factor to get required overall smoothing.
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
                db = _data_list[i-fc:i+fc+1]
                nfc = fmas+1
                ma_data.append(sum(db)/nfc)
            else:
                db = _data_list[i-(ds-i-1):]
                nfc = len(db)
                ma_data.append(sum(db)/nfc)
    
    return ma_data

def S_downsample(_data_list, _factor=1):
    """
    Returns a two dimensional data set with a reduced number of samples.
    Use the sample skipping factor to get required result, the factor tells how many samples to skip for one data sample.
    """

    ds_data = []
    ds = len(_data_list)
    skip_count = _factor

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

    if _npoints < 3:
        return [_point1, _point2]

    points = []

    step = 1 / (_npoints-1)
    t = 0
    d = _point2[0] - _point1[0]

    while t <= 1:
        x = _point1[0] + (d*t)
        points.append((x,(_point1[1]*(1-t))+(_point2[1]*t)))
        t += step

    return points


def S_upsample(_data_list, _factor=1, _smooth=False):
    """
    Returns a two dimensional data set with an increased number of samples.
    The factor tells how many samples to add for one data sample where the smooth is to use smooth cosine interpolation for added samples.
    """

    ds_data = []
    ds = len(_data_list)
    inter_f = None
    smoothing_f = 3

    if _smooth == True:
        inter_f = S_cosine_function
        smoothing_f = 4
    else:
        inter_f = S_linear_function

    for i in range(ds - 1):

        inter = inter_f((_data_list[i][0], _data_list[i][1]), (_data_list[i + 1][0], _data_list[i + 1][1]), _factor + smoothing_f)
        irs = len(inter)

        for ii in range(irs - 1):
            ds_data.append((inter[ii][0], inter[ii][1]))

    ds_data.append((_data_list[-1][0], _data_list[-1][1]))

    return ds_data

def S_cosine_function(_point1, _point2, _npoints):
    """
    Returns a list of points Cosine interpolated between point1 and point2 with npoints number of points.
    """

    if _npoints < 4:
        return [_point1, _point2]

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

def S_crop_data(_data_list, _max, _min):
    """
    Returns a filtered data where values are discarded according to max, min limits.
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

def S_uniform_spread(_data_list, _nsamples):
    """
    Returns a uniformly spread data samples where samples size is fixed by nsamples.
    """
    u_data = []

    ds = len(_data_list)
    d = _data_list[-1][0] - _data_list[0][0]
    sl = (d/_nsamples)
    dc = 0

    u_data.append((_data_list[0][0], _data_list[0][1]))

    for i in range(1, _nsamples):

        x = i * sl

        for ii in range(dc, ds):
            if _data_list[ii][0] >= x:
                dc = ii
                break

        d = _data_list[dc][0] - _data_list[dc-1][0]
        t = (x -  _data_list[dc-1][0]) / d

        u_data.append((x, (_data_list[dc-1][1] * (1 - t)) + (_data_list[dc][1] * t)))

    u_data.append((_data_list[-1][0], _data_list[-1][1]))

    return u_data

def S_smooth_data(_data_list, _smoothing=1):
    """
    Returns data samples with smooth profiles applied in between the sample points.
    """

    s_data = []
    ds = len(_data_list)

    for i in range(ds-1):

        inter = S_cosine_function((_data_list[i][0], _data_list[i][1]), (_data_list[i+1][0], _data_list[i+1][1]), _npoints=_smoothing+4)
        irs = len(inter)

        for ii in range(irs - 1):
            s_data.append((inter[ii][0], inter[ii][1]))

    s_data.append((_data_list[-1][0], _data_list[-1][1]))

    return s_data

def S_adjust_phase(_data_list, _transform):
    """
    Returns data samples where the phase is moved by transform amount.
    """

    a_data = []
    ds = len(_data_list)

    for i in range(ds):
        a_data.append((_data_list[i][0]+_transform, _data_list[i][1]))

    return a_data

def S_scale_data(_data_list, _factor):
    """
    Returns data samples where y axis values are scaled by the factor.
    """

    s_data = []
    ds = len(_data_list)

    for i in range(ds):
        s_data.append(_data_list[i]*_factor)

    return s_data

def S_change_amplitude(_data_list, _amount):
    """
    Returns data samples where values are either increased or decreased by the amount.
    """

    s_data = []
    ds = len(_data_list)

    amount = abs(_amount)


    for i in range(ds):
        if _amount > 0:
            if _data_list[i] > 0:
                s_data.append(_data_list[i] + amount)
            elif _data_list[i] < 0:
                s_data.append(_data_list[i] - amount)
            else:
                s_data.append(_data_list[i])
        elif _amount < 0:
            if _data_list[i] > 0:
                s_data.append(_data_list[i] - amount)
            elif _data_list[i] < 0:
                s_data.append(_data_list[i] + amount)
            else:
                s_data.append(_data_list[i])

    return s_data

def S_shift_data(_data_list, _transform):
    """
    Returns data samples where values are translated by transform amount.
    """

    s_data = []
    ds = len(_data_list)

    for i in range(ds):
        s_data.append(_data_list[i][1]+_transform)

    return s_data

def S_convolute_data(_data_list, _transformer):
    """
    Returns new data samples where values are transformed by transformer values.
    """

    c_data = []
    ds = len(_data_list)
    ts = len(_transformer)

    if ds != ts:
        return []

    for i in range(ds):
        c_data.append(_data_list[i] + _transformer[i])

    return c_data

def S_invert_data(_data_list):
    """
    Returns data samples where values are inverted.
    """

    i_data = []
    ds = len(_data_list)

    for i in range(ds):
        i_data.append(-1*_data_list[i])

    return i_data

def S_inverse_data(_data_list, _infinity_value='inf'):
    """
    Returns data samples where values are inversely proporsional.
    """

    i_data = []
    ds = len(_data_list)

    for i in range(ds):

        if _data_list[i] == 0:

            i_data.append(_infinity_value)

        if _data_list[i] == _infinity_value:

            i_data.append(0.0)

        else:

            i_data.append(1/_data_list[i])

    return i_data

def S_translate_data(_data_list, _transform_x, _transform_y):
    """
    Returns data samples where data points are translated by transform amount.
    """

    t_data = []
    ds = len(_data_list)

    for i in range(ds):
        t_data.append((_data_list[i][0]+_transform_x, _data_list[i][1]+_transform_y))

    return t_data

def S_translate_to_positive_axis(_data_list):
    """
    Returns data samples where data points are translated to positive X and Y axes.
    """

    x_val, y_val = list(zip(*_data_list))

    x_transform = abs(min(x_val))
    y_transform = abs(min(y_val))

    return S_translate_data(_data_list, x_transform, y_transform)

def S_envelope_filter(_data_list, _upper=True, _level=0.001, _step=1):
    """
    Returns the envelope of data samples.
    Use the level parameter to tweak envelope detection and set upper to False to get lower envelope.
    """

    peaks = mining.S_get_all_peaks(_data_list, _level=_level, _step=_step, _valley=not _upper)

    e_data = []

    for i in range(0, len(peaks)-1):
        samples = S_linear_function((peaks[i][1], peaks[i][0]), (peaks[i+1][1], peaks[i+1][0]),  peaks[i+1][1]- peaks[i][1])
        for j in range(len(samples)):
            e_data.append(samples[j][1])

    samples = S_linear_function((peaks[-2][1], peaks[-2][0]), (peaks[-1][1], peaks[-1][0]), peaks[-1][1] - peaks[-2][1])
    for j in range(len(samples)):
        e_data.append(samples[j][1])

    return e_data


def S_envelope_approximate_filter(_data_list, _smoothing=1, _upper=True):
    """
    Returns the envelope of data samples by combining standard deviation to moving average data.
    Use the smoothing parameter to tweak envelope amplitude.
    """

    ea_data = []
    ds = len(_data_list)
    s = _smoothing
    mas = int((ds * 0.02) * s)
    fc = int(mas / 2)
    fmas = fc * 2

    for i in range(ds):
        if i < fc:
            db = _data_list[:i + i + 1]
            nfc = len(db)
            ma = sum(db) / nfc
            sd = mining.S_standard_deviation(db)
            if _data_list[i] > 0:
                if _upper == True:
                    ea_data.append(ma+sd)
                else:
                    ea_data.append(ma-sd)
            elif _data_list[i] < 0:
                if _upper == True:
                    ea_data.append(ma-sd)
                else:
                    ea_data.append(ma+sd)
            else:
                ea_data.append(ma)

        elif i >= fc:
            if i < (ds - fc):
                db = _data_list[i - fc:i + fc + 1]
                nfc = fmas + 1
                ma = sum(db) / nfc
                sd = mining.S_standard_deviation(db)
                if _data_list[i] > 0:
                    if _upper == True:
                        ea_data.append(ma + sd)
                    else:
                        ea_data.append(ma - sd)
                elif _data_list[i] < 0:
                    if _upper == True:
                        ea_data.append(ma - sd)
                    else:
                        ea_data.append(ma + sd)
                else:
                    ea_data.append(ma)
            else:
                db = _data_list[i - (ds - i - 1):]
                nfc = len(db)
                ma = sum(db) / nfc
                sd = mining.S_standard_deviation(db)
                if _data_list[i] > 0:
                    if _upper == True:
                        ea_data.append(ma + sd)
                    else:
                        ea_data.append(ma - sd)
                elif _data_list[i] < 0:
                    if _upper == True:
                        ea_data.append(ma - sd)
                    else:
                        ea_data.append(ma + sd)
                else:
                    ea_data.append(ma)

    return ea_data

def S_duplicates_filter(_data_list):
    """
    Returns data samples where duplicated data samples are replaced by interpolated values.
    """

    c_data = []

    ds = len(_data_list)

    i_points_s = []
    i_points_e = []

    sq = mining.S_find_square_floors(_data_list)

    dsq = len(sq)

    for i in sq:
        if i[1] == ds - 1:
            i_points_s.append(i[0])
            i_points_e.append(i[1])
        else:
            i_points_s.append(i[0])
            i_points_e.append(i[1] + 1)

    counter = 0

    while counter < ds:

        if counter in i_points_s:
            end = i_points_e[i_points_s.index(counter)]

            samples = S_cosine_function((counter, _data_list[counter]), (end, _data_list[end]), (end-counter)+1)
            samples_len = len(samples)

            if end in i_points_s:
                samples_len -= 1

            for j in range(samples_len):
                c_data.append(samples[j][1])
                counter += 1
        else:
            c_data.append(_data_list[counter])
            counter += 1


    return c_data

def S_generate_interpolate_points(_data_list):
    """
    Returns new data samples by interpolating data samples
    """

    g_data = []

    ds = len(_data_list)

    for i in range(ds-1):
        samples = S_cosine_function((_data_list[i][0], _data_list[i][1]), (_data_list[i+1][0], _data_list[i+1][1]), (_data_list[i+1][0] - _data_list[i][0]) + 1)
        samples_len = len(samples)

        for j in range(samples_len-1):
            g_data.append((samples[j][0], samples[j][1]))


    g_data.append(_data_list[-1])

    return g_data


def S_generate_triangle_signal(_wave_length, _amplitude, _nwaves):
    """
    Returns triangle signal data samples where nwaves defines number of full waves
    """

    _qspace = 5

    q_length = 0

    if _wave_length > 7:
        q_length = int(_wave_length / 4) + 1
    else:
        q_length = 3

    g_data = []

    for i in range(_nwaves):

        cursor = i*_qspace
        qq = 2*_qspace
        qqq = 3*_qspace

        quadrant1 = S_linear_function((cursor, 0), (cursor+_qspace, _amplitude), q_length)
        quadrant2 = S_linear_function((cursor+_qspace, _amplitude), (cursor+qq, 0), q_length)
        quadrant3 = S_linear_function((cursor+qq, 0), (cursor+qqq, -_amplitude), q_length)
        quadrant4 = S_linear_function((cursor+qqq, -_amplitude), (cursor+(4*_qspace), 0), q_length)


        fwave = quadrant1[:-1]+quadrant2[:-1]+quadrant3[:-1]+quadrant4[:-1]

        x_val, y_val = zip(*fwave)

        g_data.extend(y_val)

    g_data.append(0)

    return g_data

def S_gradient_filter(_data_list, _diff=0.1):
    """
    Returns data samples where points with same gradient are discarded.
    """

    g_data = []

    ds = len(_data_list)

    pgrad = 0

    for i in range(ds-1):

        grad = (_data_list[i+1][1]-_data_list[i][1])/(_data_list[i+1][0]-_data_list[i][0])

        d = abs(pgrad-grad)
        lv = abs(max(pgrad, grad))

        if lv == 0:
            g_data.append(_data_list[i])
            pgrad = grad
            continue

        if not ((d / lv) <= _diff):
            g_data.append(_data_list[i])
            pgrad = grad

    g_data.append(_data_list[-1])

    return g_data

def S_kalman(u, R=10, H=1.0):
    """
    Returns a new sample based on Kalman algorithm where R is noise covariance and H is a measurement scalar.
    """

    Q = 10  # initial estimated covariance
    P = 0   # initial error covariance
    u_new = 0
