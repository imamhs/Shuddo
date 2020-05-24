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