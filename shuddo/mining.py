# Copyright (c) 2020, Md Imam Hossain (emamhd at gmail dot com)
# see LICENSE.txt for details

"""
Data mining functions
"""

from math import isclose, hypot, pi

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
        return (peak_value, _cursor + (peak_width / 2), peak_width, peak_start_location, cursor-1)
    finally:
        return (peak_value, _cursor + (peak_width/2), peak_width, peak_start_location, cursor)
        
        
def S_get_all_peaks(_data_list, _level=0.5, _step=1):
    """
    Returns all the peaks available in the data based on height differences between points next to each other as set by level, step defines number of points to skip for checking the level difference
    """

    ds = len(_data_list)
    
    peaks = []
    data_point = []
    peak_start_location = -1 # negative for indicating not set
    peak_end_location = -1 # negative for indicating not set
    cursor = 0

    while cursor < (ds - _step):

        level_change = _data_list[cursor + _step] - _data_list[cursor]

        if level_change > _level and peak_start_location == -1:
            peak_start_location = cursor
        elif level_change < 0 and peak_end_location == -1 and peak_start_location >= 0:
            peak_end_location = cursor

        if peak_start_location >= 0:
            data_point.append(_data_list[cursor])

        if peak_start_location >= 0 and peak_end_location >= 0:
            if len(data_point) > 2:
                half_width = (peak_end_location - peak_start_location)
                peak_width = half_width * 2
                peak_location = peak_end_location
                peak_valley = min(data_point)
                peak_value = max(data_point)
                peak_height = abs(peak_value - peak_valley)
                peaks.append((peak_value, peak_location, peak_width, peak_height, peak_start_location))
                data_point.clear()
                peak_start_location = -1
                peak_end_location = -1

        cursor += _step
    
    return peaks

def S_is_neighbour(_data_list, _sample, _similarity=0.8):
    """
    Checks if provided a sample is in close proximity for a given data samples
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
    Finds the centroid of a given two dimensional data samples cluster
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
    Finds the radius of a given two dimensional data samples cluster
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
    Finds the clusters present for given two dimensional scattered data samples
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
    Finds the groups present for given data samples
    Groups will be divided based on value difference as set by level
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

