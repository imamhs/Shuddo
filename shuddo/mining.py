# Copyright (c) 2020, Md Imam Hossain (emamhd at gmail dot com)
# see LICENSE.txt for details

"""
Data mining functions
"""

from math import isclose, hypot, pi, floor

def S_get_peaks_valleys(_data_list, _level=0.01, _distance=1):
    """
    Returns all the peaks and valleys present in the data samples
    The level parameter defines the minimum level change required between consecutive peaks and valleys
    The distance parameter defines the minimum distance required between consecutive peaks and valleys
    """

    ds = len(_data_list)

    peaksvalleys = []

    moving_p = (0, 0)

    dirup = True

    for i in range(0, ds-1):
        diff = _data_list[i+1] - _data_list[i]
        if diff > 0:
            if dirup == False:
                if abs(_data_list[i] - moving_p[1]) > _level and abs(i - moving_p[0]) > _distance:
                    peaksvalleys.append((i, _data_list[i]))
                    moving_p = (i, _data_list[i])
            dirup = True
        elif diff < 0:
            if dirup == True:
                if abs(_data_list[i] - moving_p[1]) > _level and abs(i - moving_p[0]) > _distance:
                    peaksvalleys.append((i, _data_list[i]))
                    moving_p = (i, _data_list[i])
            dirup = False

    return peaksvalleys


def S_get_peak(_data_list, _cursor=0, _base_line=0):
    """
    Returns the features of a first positive or negative peak from the data, define horizontal separation for the peak detection by setting the base line, define detection starting location by setting the cursor
    """

    data_point = []
    peak_value = 0
    cursor = _cursor
    peak_start_location = -1 # negative for indicating not set
    peak_width = 0

    try:
        while _data_list[cursor] == _base_line:
            cursor += 1
    except IndexError:
        return (0, -1, -1, -1, -1)

    try:
        if _data_list[cursor] < _base_line:
            peak_start_location = cursor
            while _data_list[cursor] < _base_line:
                data_point.append(_data_list[cursor])
                cursor += 1
            peak_width = cursor - peak_start_location
            peak_value = min(data_point)
        elif _data_list[cursor] > _base_line:
            peak_start_location = cursor
            while _data_list[cursor] > _base_line:
                data_point.append(_data_list[cursor])
                cursor += 1
            peak_width = cursor - peak_start_location
            peak_value = max(data_point)

    except IndexError:
        if _data_list[cursor-1] < _base_line:
            peak_value = min(data_point)
        elif _data_list[cursor-1] > _base_line:
            peak_value = max(data_point)
        peak_width = (cursor-1) - peak_start_location
        return (_cursor + (peak_width / 2), peak_value, peak_width, peak_start_location, cursor-1)
    finally:
        return (_cursor + (peak_width/2), peak_value, peak_width, peak_start_location, cursor)
        
        
def S_get_all_peaks(_data_list, _level=0.5, _step=1, _valley=False):
    """
    Returns all the peaks available in the data based on height differences between points next to each
    other as set by level where step defines the number of points to skip for checking the level difference.
    """

    ds = len(_data_list)
    
    peaks = []
    data_point = []
    peak_start_location = -1 # negative for indicating not set
    peak_end_location = -1 # negative for indicating not set
    cursor = 0

    while cursor < (ds - _step):

        level_change = _data_list[cursor + _step] - _data_list[cursor]

        if _valley == False:
            if level_change > _level and peak_start_location == -1:
                peak_start_location = cursor
            elif level_change < 0 and peak_end_location == -1 and peak_start_location >= 0:
                peak_end_location = cursor
        else:
            if level_change < _level and peak_start_location == -1:
                peak_start_location = cursor
            elif level_change > 0 and peak_end_location == -1 and peak_start_location >= 0:
                peak_end_location = cursor

        if peak_start_location >= 0:
            data_point.append(_data_list[cursor])

        if _valley == False:

            if peak_start_location >= 0 and peak_end_location >= 0:
                if len(data_point) > 2:
                    half_width = (peak_end_location - peak_start_location)
                    peak_width = half_width * 2
                    peak_location = peak_end_location
                    peak_valley = min(data_point)
                    peak_value = max(data_point)
                    peak_height = abs(peak_value - peak_valley)
                    peaks.append((peak_location, peak_value, peak_width, peak_height, peak_start_location))
                    data_point.clear()
                    peak_start_location = -1
                    peak_end_location = -1

        else:

            if peak_start_location >= 0 and peak_end_location >= 0:
                if len(data_point) > 2:
                    half_width = (peak_end_location - peak_start_location)
                    peak_width = half_width * 2
                    peak_location = peak_end_location
                    peak_valley = max(data_point)
                    peak_value = min(data_point)
                    peak_height = abs(peak_value - peak_valley)
                    peaks.append((peak_location, peak_value, peak_width, peak_height, peak_start_location))
                    data_point.clear()
                    peak_start_location = -1
                    peak_end_location = -1


        cursor += _step
    
    return peaks

def S_is_neighbour(_data_list, _sample, _similarity=0.8):
    """
    Checks if provided sample is in close proximity for a given data samples.
    """

    ds = len(_data_list)

    for i in range(ds):

        x_sim = False
        y_sim = False

        if isclose(_data_list[i][0], _sample[0], abs_tol=_similarity):
            x_sim = True

        if isclose(_data_list[i][1], _sample[1], abs_tol=_similarity):
            y_sim = True

        if x_sim and y_sim:
            return True

    return False

def S_get_cluster_centroid(_data_list):
    """
    Finds the centroid of a given two dimensional data samples cluster.
    """
    ds = len(_data_list)

    if ds > 0:
        x_sum = 0
        y_sum = 0
        for i in _data_list:
            x_sum += i[0]
            y_sum += i[1]
        return (x_sum / ds, y_sum / ds)

def S_get_cluster_radius(_data_list, _center):
    """
    Finds the radius of a given two dimensional data samples cluster.
    """

    ds = len(_data_list)

    if ds > 0:
        dists = []
        for i in _data_list:
            dists.append(hypot(i[0] - _center[0], i[1] - _center[1]))
        radius = max(dists)
        if radius == 0:
            return 0
        return (radius, ds/(pi*(radius**2)))

def S_get_clusters(_data_list, _similarity=0.8):
    """
    Finds the clusters present for given two dimensional scattered data samples.
    """

    clusters = []
    centroids = []
    ds = len(_data_list)

    for i in range(ds):

        if len(clusters) == 0:
            clusters.append([(_data_list[i][0], _data_list[i][1])])
        else:
            neighbour = False
            for ii in range(len(clusters)):
                if S_is_neighbour(clusters[ii], _data_list[i], _similarity=_similarity):
                    clusters[ii].append((_data_list[i][0], _data_list[i][1]))
                    neighbour = True
                    break
            if neighbour == False:
                clusters.append([(_data_list[i][0], _data_list[i][1])])

    for i in clusters:
        centroids.append(S_get_cluster_centroid(i))

    return list(zip(clusters, centroids))

def S_get_histogram(_data_list, _level=0.1):
    """
    Finds the groups present for given data samples.
    Groups will be divided based on value difference as set by level.
    """

    s_data = sorted(_data_list)

    bins = []
    bin_cursor = 0
    averages = []
    counts = []
    comparator = s_data[0]

    bins.append([s_data[0]])

    ds = len(s_data)

    for i in range(1, ds):
        if s_data[i] > (comparator + _level):
            bin_cursor += 1
            bins.append([s_data[i]])
            comparator = s_data[i]
            continue
        else:
            bins[bin_cursor].append(s_data[i])

    for i in bins:
        sz = len(i)
        averages.append(sum(i)/sz)
        counts.append(sz)

    return list(zip(averages, counts))

def S_check_similarity(_data_lista, _data_listb, _band=0.1, _tolerance=5):
    """
    Checks if two data sets are similar when samples are aligned.
    Band determines difference in values and tolerance tells number of dissimilar samples to ignore.
    """

    dsa = len(_data_lista)
    dsb = len(_data_listb)

    tol = 0

    if dsa != dsb:
        return False

    for i in range(dsa):

        if isclose(_data_lista[i], _data_listb, abs_tol=_band) == False:
            tol += 1

        if tol >= _tolerance:
            return False

    return True

def S_variance(_data_list):

    ds = len(_data_list)
    mean = sum(_data_list)/ds
    average_spread = 0

    for i in _data_list:
        average_spread += (i - mean)**2

    average_spread = average_spread / ds

    return average_spread

def S_standard_deviation(_data_list):

    return (S_variance(_data_list))**(1/2)

def S_median_sample(_data_list):
    """
    Returns middle value from data set and its location.
    """

    vals = sorted(_data_list)
    ds = len(vals)
    i = floor(ds / 2)

    if ds%2 == 1:
        return (i+1, vals[i])
    else:
        return ((i+0.5),((vals[i-1] + vals[i])/2))


def S_count_leftright(_data_list, _value):
    """
    Returns the number of samples to the left and the right of a sample from a data set as expressed by value.
    """

    vals = sorted(_data_list)
    left_c = 0
    right_c = 0

    for i in vals:

        if _value > i:
            left_c += 1
        else:
            right_c += 1

    return (left_c, right_c)

def S_average_high(_data_list, _percent=0.1):
    """
    Returns average of higher value samples where the number of samples to consider is defined by percent.
    1.0 equals all samples, 0.2 equals 20 percent of samples from a data set.
    """

    if _percent > 1.0 or _percent < 0.1:
        return None

    vals = sorted(_data_list)
    ds = len(vals)
    i = floor(ds * (1-_percent))
    samples = vals[i-ds:ds]
    samples_len = len(samples)
    return sum(samples)/samples_len

def S_average_low(_data_list, _percent=0.5):
    """
    Returns average of lower value samples where the number of samples to consider is defined by percent.
    1.0 equals all samples, 0.2 equals 20 percent of samples from a data set.
    """

    if _percent > 1.0 or _percent < 0.1:
        return None

    vals = sorted(_data_list)
    ds = len(vals)
    i = floor(ds * _percent)
    samples = vals[:i]
    samples_len = len(samples)
    return sum(samples)/samples_len

def S_proximity(_data_list, _percent_similarity=0.95):
    """
    Calculates average of smallest distances between data points when samples are translated to positive values.
    Percent similarity is used for determining closeness between data points.
    Returns smallest distances along X and Y axes.
    """

    x_val, y_val = list(zip(*_data_list))
    
    x_val = sorted(x_val)
    y_val = sorted(y_val)

    x_diffs = []
    y_diffs = []

    ds = len(x_val)

    for i in range(1, ds):
        x_diff = x_val[i] - x_val[i-1]
        y_diff = y_val[i] - y_val[i-1] 
        if x_diff >= _percent_similarity:
            continue
        else:
            x_diffs.append(x_diff)

        if y_diff >= _percent_similarity:
            continue
        else:
            y_diffs.append(y_diff)


    x_proximity = sum(x_diffs)/len(x_diffs)
    y_proximity = sum(y_diffs)/len(y_diffs)

    return (x_proximity, y_proximity)

def S_find_range(_data_list):
    """
    Returns smallest and largest values in the one dimensional dataset.
    """
    return (min(_data_list), max(_data_list))

def S_covariance(_data_list):
    """
    Returns covariance of two dimensional data samples.
    """

    ds = len(_data_list)

    cv = 0

    if ds > 0:
        for i in _data_list:
            cv += (i[0] * i[1])

        return (cv/ds)
    else:
        return None

def S_find_gradients(_data_list):
    """
    Returns all gradients from the data samples.
    """

    g_data = []

    ds = len(_data_list)

    for i in range(1, ds):

        g = _data_list[i] - _data_list[i-1]

        if g != 0:
            g_data.append((g, i))

    return g_data

def S_find_square_floors(_data_list):
    """
    Returns locations of values which do not change.
    """

    s_data = []

    ds = len(_data_list)

    pd = _data_list[0]

    start = end = -1

    for i in range(1, ds):

        if pd == _data_list[i]:
            if start == -1:
                start = i - 1
            if start != -1 and i == ds-1:
                s_data.append((start, ds-1))
        else:
            if start != -1:
                end = i - 1
                s_data.append((start, end))
                start = end = -1

        pd = _data_list[i]

    return s_data