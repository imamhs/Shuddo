# Copyright (c) 2019-2022, Md Imam Hossain (emamhd at gmail dot com)
# see LICENSE.txt for details

"""
Data filtering functions
"""

from random import random
from math import pi, cos, hypot
from shuddo import mining

def S_moving_average_filter(_data_list, _smoothing=1):
    """
    Returns moving average data without data lag.
    Use the smoothing factor to get required overall smoothing where the smoothing factor is greater than zero.
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

def S_downsample_data(_data_list, _factor=1):
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


def S_upsample_data(_data_list, _factor=1, _smooth=False):
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

def S_crop_values(_data_list, _max, _min):
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

def S_uniform_spread_data(_data_list, _nsamples):
    """
    Returns a uniformly spread data samples where samples size is fixed by nsamples.
    """
    u_data = []

    ds = len(_data_list)
    d = _data_list[-1][0] - _data_list[0][0]

    if d == 0:
        return None

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

def S_adjust_phase_data(_data_list, _transform):
    """
    Returns data samples where the phase is moved by transform amount.
    """

    a_data = []
    ds = len(_data_list)

    for i in range(ds):
        a_data.append((_data_list[i][0]+_transform, _data_list[i][1]))

    return a_data

def S_scale_values(_data_list, _factor):
    """
    Returns data samples where values are scaled by the factor.
    """

    s_data = []
    ds = len(_data_list)

    for i in range(ds):
        s_data.append(_data_list[i]*_factor)

    return s_data

def S_change_amplitude_values(_data_list, _amount):
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
    Returns data samples where y-axis values are translated by transform amount.
    """

    s_data = []
    ds = len(_data_list)

    for i in range(ds):
        s_data.append(_data_list[i][1]+_transform)

    return s_data

def S_convolute_values(_data_list, _transformer):
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

def S_invert_values(_data_list):
    """
    Returns data samples where values are inverted.
    """

    i_data = []
    ds = len(_data_list)

    for i in range(ds):
        i_data.append(-1*_data_list[i])

    return i_data

def S_inverse_values(_data_list, _infinity_value='inf'):
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

def S_translate_values(_data_list, _transform_amount):
    """
    Returns data samples where samples are translated by transform amount.
    """

    t_data = []
    ds = len(_data_list)

    for i in range(ds):
        t_data.append(_data_list[i]+_transform_amount)

    return t_data

def S_translate_data(_data_list, _transform_x, _transform_y):
    """
    Returns data samples where data points are translated by transform amount.
    """

    t_data = []
    ds = len(_data_list)

    for i in range(ds):
        t_data.append((_data_list[i][0]+_transform_x, _data_list[i][1]+_transform_y))

    return t_data

def S_translate_to_positive_quadrant_data(_data_list):
    """
    Returns data samples where data points are translated to positive X and Y axes.
    """

    x_val, y_val = list(zip(*_data_list))

    x_transform = abs(min(x_val))
    y_transform = abs(min(y_val))

    return S_translate_data(_data_list, x_transform, y_transform)

def S_translate_to_positive_axis_values(_data_list):
    """
    Returns data samples where data points are translated to positive axis.
    """

    transform = abs(min(_data_list))

    return S_translate_values(_data_list, transform)

def S_envelope_filter(_data_list, _upper=True, _level=0.001, _step=1):
    """
    Returns the envelope of data samples.
    Use the level parameter to tweak envelope detection and set upper to False to get lower envelope.
    """

    peaks = mining.S_get_all_peaks_values(_data_list, _level=_level, _step=_step, _valley=not _upper)

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
            sd = mining.S_standard_deviation_values(db)
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
                sd = mining.S_standard_deviation_values(db)
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
                sd = mining.S_standard_deviation_values(db)
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

    sq = mining.S_find_square_floors_values(_data_list)

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

def S_generate_interpolate_points_data(_data_list):
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

def S_generate_triangle_signal_values(_wave_length, _amplitude, _nwaves):
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

        grad = _data_list[i+1]-_data_list[i]

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

def S_kalman_filter(_data_list, R=10, H=1.0, Q=10, P=0, u_new=0):
    """
    Returns a new data samples based on Kalman algorithm where R is measured noise covariance and H is a measurement scalar.
    Q is initial estimated covariance (process variance)
    P is initial error covariance
    u_new is initial estimated value
    """

    k_data = []

    ds = len(_data_list)

    for i in range(ds):
        K = (P * H) / (((H ** 2) * P) + R)  # update Kalman gain
        u_new += (K * (_data_list[i] - (H * u_new)))  # estimated value
        P = ((1 - (K * H)) * P) + Q  # update error covariance
        k_data.append(u_new)

    return k_data

def S_outlier_filter(_data_list, _factor=4):
    """
    Returns data samples where data samples significantly greater than median are discarded.
    """

    n_data = []

    ds = len(_data_list)

    p_d = S_translate_to_positive_axis_values(_data_list)

    m_d = mining.S_median_sample_values(p_d)

    b_v = m_d[1] * _factor  # boundary value for finding outliers

    for i in range(ds):
        if p_d[i] <= b_v:
            n_data.append(_data_list[i])

    return n_data

def S_difference_values(_data_lista, _data_listb):
    """
    Returns new data samples where values are transformed by transformer values.
    """

    d_data = []
    dsa = len(_data_lista)
    dsb = len(_data_listb)

    if dsa != dsb:
        return []

    for i in range(dsa):
        d_data.append(_data_lista[i] - _data_listb[i])

    return d_data

def S_self_operate_values(_data_list, _step=1, _operation=1):
    """
    Apply arithmetic operation on data samples themselves
    step parameter is used to define how many data points to skip for calculating accelerated values
    _operation parameter is used for selecting one of the operations (1: addition, 2: subtraction, 3: multiplication, 4: division, 5: modulus)
    """

    a_data = []
    ds = len(_data_list)
    cursor = _step
    found_data = False
    res = 0

    if ds == 0:
        return []

    for i in range(_step):
        a_data.append(0.0)

    while cursor < (ds):

        if _operation == 1:
            res = _data_list[cursor] + _data_list[cursor - 1]
        elif _operation == 2:
            res = _data_list[cursor] - _data_list[cursor - 1]
        elif _operation == 3:
            res = _data_list[cursor] * _data_list[cursor - 1]
        elif _operation == 4:
            res = _data_list[cursor] / _data_list[cursor - 1]
        elif _operation == 5:
            res = _data_list[cursor] % _data_list[cursor - 1]
        else:
            res = 0

        if found_data == True:
            a_data.append(res)
        else:
            if _data_list[cursor] != 0:
                found_data = True
                if _data_list[cursor] != 0 and _data_list[cursor-1] != 0:
                    a_data.append(res)
                else:
                    a_data.append(0.0)
            else:
                a_data.append(0.0)

        cursor += _step

    return a_data

def S_aggravate_filter(_data_list, _iteration=1):
    """
    Returns data samples where original data samples are aggravated for _iteration number of times
    and then factored to scale to match original data samples amplitude
    """

    if _iteration < 1:
        return []

    a_data = []

    ds_max = abs(max(_data_list))
    ds_min = abs(min(_data_list))

    scale_factor_d = ds_max

    if ds_min > scale_factor_d:
        scale_factor_d = ds_min

    a_data = S_self_operate_values(_data_list)

    for i in range(_iteration-1):
        a_data = S_self_operate_values(a_data)

    ds_max = abs(max(a_data))
    ds_min = abs(min(a_data))

    scale_factor_ad = ds_max

    if ds_min > scale_factor_ad:
        scale_factor_ad = ds_min

    scale_factor = scale_factor_d / scale_factor_ad

    a_data = [i * scale_factor for i in a_data]

    return a_data

def S_round_filter(_data_list, _factor=2):
    """
    Round up data samples using round method
    """

    r_data = []

    r_data = [round(i,_factor) for i in _data_list]

    return r_data

def S_cumulative_sums_values(_data_list, _initial_value=0):
    """
    Integrate data samples to produce new data samples
    """

    ds = len(_data_list)

    if ds == 0:
        return []

    i_data = []

    cursor = 0

    for i in range(ds):
        if _data_list[i] == 0:
            i_data.append(0.0)
        else:
            break
        cursor += 1

    if cursor != 0:
        if _initial_value != 0:
            i_data[cursor - 1] = _initial_value
    else:
        if _initial_value != 0:
            i_data.append(_initial_value)
        else:
            i_data.append(_data_list[0])
        cursor += 1

    for i in range(cursor, ds):
        i_data.append(_data_list[i]+i_data[i-1])

    return i_data

def S_acceleration_filter(_data_list, _smoothing=1, _iteration=1):
    """
    Returns data samples where original data samples are accelerated for _iteration number of times
    and then smoothed to get smoothing effect on the data samples
    """

    ds = len(_data_list)

    if ds == 0:
        return []

    i_values = []

    i_values.append(mining.S_find_first_nonzero_values(_data_list))
    a_data = S_self_operate_values(_data_list, _operation=2)

    for i in range(_iteration-1):
        i_values.append(mining.S_find_first_nonzero_values(a_data))
        a_data = S_self_operate_values(a_data, _operation=2)

    a_data = S_moving_average_filter(a_data, _smoothing=_smoothing)

    for i in reversed(i_values):
        a_data = S_cumulative_sums_values(a_data, _initial_value=i)

    return a_data
    
def S_linear_model_data(_data_list):
    """
    Returns linear regression model of data samples
    """
    
    ds = len(_data_list)
    
    if ds < 2:
        return -1

    r_data = []

    x_val, y_val = list(zip(*_data_list))
    
    x_val_sum = sum(x_val)
    y_val_sum = sum(y_val)
    
    x_val_power = [i*i for i in x_val]
    x_val_power_sum = sum(x_val_power)
    x_y_val = [x*y for x, y in _data_list]
    x_y_val_sum = sum(x_y_val)
        
    a = ((y_val_sum*x_val_power_sum)-(x_val_sum*x_y_val_sum))/((ds*x_val_power_sum) - (x_val_sum**2))
    b = ((ds*x_y_val_sum) - (x_val_sum*y_val_sum))/((ds*x_val_power_sum) - (x_val_sum**2))
    
    for i in x_val:
        r_data.append((i, a+(b*i)))
        
    return r_data
    
def S_linear_model_window_data(_data_list, _window=10):
    """
    Returns linear regression model of data samples using window size steps
    """
    
    ds = len(_data_list)
    
    if ds < 2:
        return -1
    
    r_data  = []
    
    cursor = 0
    
    while cursor < (ds):
    
        lr = S_linear_model_data(_data_list[cursor:cursor+_window])
        
        for i in lr:

            r_data.append(i)

        cursor += _window

    return r_data    

def S_cluster_outlier_filter(_data_list, _similarity=0.8):
    """
    Returns data samples where a data sample outside a cluster is discarded
    """

    ds = len(_data_list)

    if ds < 2:
        return -1

    n_data = []
    d_data = []

    c = mining.S_get_clusters_data(_data_list, _similarity)

    for i in c:
        if len(i[0]) > 1:
            for ii in i[0]:
                n_data.append(ii)
        elif len(i[0]) == 1:
            d_data.append(i[0][0])

    return (n_data, d_data)

def S_generate_noise_signal_values(_samples, _amplitude_factor):
    """
    Returns noise signal data samples where _amplitude_factor defines the peak magnitude in the signal
    """

    n_data = []

    for i in range(_samples):

        n_data.append(random()*(_amplitude_factor*2))

    return S_translate_values(n_data, -1 * _amplitude_factor)

def S_envelope_approximate_filter_rms(_data_list, _smoothing=1, _upper=True):
    """
    Returns the envelope of data samples by combining RMS to moving average data.
    Use the smoothing parameter to tweak envelope amplitude.
    """

    transform = abs(min(_data_list))

    p_val = S_translate_values(_data_list, transform)

    ea_data = []
    ds = len(p_val)
    s = _smoothing
    mas = int((ds * 0.02) * s)
    fc = int(mas / 2)
    fmas = fc * 2

    for i in range(ds):
        if i < fc:
            db = p_val[:i + i + 1]
            nfc = len(db)
            ma = sum(db) / nfc
            sd = mining.S_find_rms_values(db)
            if p_val[i] > 0:
                if _upper == True:
                    ea_data.append(ma+sd)
                else:
                    ea_data.append(ma-sd)
            elif p_val[i] < 0:
                if _upper == True:
                    ea_data.append(ma-sd)
                else:
                    ea_data.append(ma+sd)
            else:
                ea_data.append(ma)

        elif i >= fc:
            if i < (ds - fc):
                db = p_val[i - fc:i + fc + 1]
                nfc = fmas + 1
                ma = sum(db) / nfc
                sd = mining.S_find_rms_values(db)
                if p_val[i] > 0:
                    if _upper == True:
                        ea_data.append(ma + sd)
                    else:
                        ea_data.append(ma - sd)
                elif p_val[i] < 0:
                    if _upper == True:
                        ea_data.append(ma - sd)
                    else:
                        ea_data.append(ma + sd)
                else:
                    ea_data.append(ma)
            else:
                db = p_val[i - (ds - i - 1):]
                nfc = len(db)
                ma = sum(db) / nfc
                sd = mining.S_find_rms_values(db)
                if p_val[i] > 0:
                    if _upper == True:
                        ea_data.append(ma + sd)
                    else:
                        ea_data.append(ma - sd)
                elif p_val[i] < 0:
                    if _upper == True:
                        ea_data.append(ma - sd)
                    else:
                        ea_data.append(ma + sd)
                else:
                    ea_data.append(ma)

    return S_translate_values(ea_data, -1*transform)

def S_dissimilar_filter_data(_data_lista, _data_listb, _radius=0.1):
    """
    Discard values from data set where not similar when samples are aligned
    """

    dsa = len(_data_lista)
    dsb = len(_data_listb)

    if dsa != dsb:
        return None

    d_data = []

    for i in range(dsa):

        if hypot(_data_lista[i][0] - _data_listb[i][0], _data_lista[i][1] - _data_listb[i][1]) > _radius:
            if i == 0:
                d_data.append(_data_listb[i])
            elif i == dsa-1:
                d_data.append(_data_listb[i])
            else:
                d_data.append(S_linear_function(_data_lista[i-1], _data_lista[i+1], 3)[1])
        else:
            d_data.append(_data_lista[i])

    return d_data