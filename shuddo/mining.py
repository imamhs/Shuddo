# Copyright (c) 2020, Md Imam Hossain (emamhd at gmail dot com)
# see LICENSE.txt for details

"""
Data mining functions
"""

def S_get_peak(_data_list, _cursor=0, _base_line=0):
    """
    Returns the features of a first positive or negative peak from the data, define horizontal separation for the peak detection by setting the base line, define detection starting location by setting the cursor
    """

    data_point = []
    peak_value = 0
    peak_location = 0
    cursor = _cursor
    peak_width = 0

    try:
        while _data_list[cursor] == _base_line:
            cursor += 1
    except IndexError:
        return (0, -1, -1, -1)

    try:
        if _data_list[cursor] < _base_line:
            while _data_list[cursor] < _base_line:
                data_point.append(_data_list[cursor])
                cursor += 1
                peak_width += 1
            peak_value = min(data_point)
        elif _data_list[cursor] > _base_line:
            while _data_list[cursor] > _base_line:
                data_point.append(_data_list[cursor])
                cursor += 1
                peak_width += 1
            peak_value = max(data_point)

    except IndexError:
        if _data_list[cursor-1] < _base_line:
            peak_value = min(data_point)
        elif _data_list[cursor-1] > _base_line:
            peak_value = max(data_point)
        return (peak_value, _cursor + (peak_width / 2), peak_width, cursor)
    finally:
        return (peak_value, _cursor+(peak_width/2), peak_width, cursor)
        
        
def S_get_all_peaks(_data_list, _gradient=5, _noise=5):
    
    
    pass
    ds_data = []   
    ds = len(_data_list)
    skip_count = 0
    
    peaks = []
    data_point = []
    peak_value = 0
    peak_location = -1
    peak_width = 0
    level_change = 0
    peak_start_location = -1
    peak_end_location = -1
    
    for i in range(ds-1):
        
        if skip_count < _noise:
            if peak_start_location > 0:
                data_point.append(_data_list[i])
            skip_count += 1
        else:
            level_change = _data_list[i+1] - _data_list[i]
            if level_change > _gradient and peak_start_location == -1:
                peak_start_location = i
            elif level_change < -_gradient and peak_end_location == -1:
                peak_end_location = i
            
            if peak_start_location > 0 and peak_end_location > 0:
                if len(data_point) > 2:
                    peak_width = peak_end_location - peak_start_location
                    peak_location = peak_start_location + (peak_width/2)
                    peak_value = max(data_point)
                    peaks.append((peak_value, peak_location, peak_width))
                    data_point.clear()
                    peak_start_location = -1
                    peak_end_location = -1
                    peak_width = 0
                    peak_location = -1
                    peak_value = -1
                
            skip_count = 0
    
    return peaks