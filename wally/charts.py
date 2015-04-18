import os
import sys
import hashlib

from GChartWrapper import Line
from GChartWrapper import constants
from GChartWrapper import VerticalBarGroup

from config import cfg_dict


# Patch MARKER constant
constants.MARKERS += 'E'
sys.modules['GChartWrapper.GChart'].MARKERS += 'E'


COLORS = ["1569C7", "81D8D0", "307D7E", "5CB3FF", "0040FF", "81DAF5"]
constants.MARKERS += 'E'  # append E marker to available markers


def get_top_top_dir(path):
    top_top_dir = os.path.dirname(os.path.dirname(path))
    return path[len(top_top_dir) + 1:]


def render_vertical_bar(title, legend, bars_data, bars_dev_top,
                        bars_dev_bottom, file_name,
                        width=700, height=400,
                        scale_x=None, scale_y=None, label_x=None,
                        label_y=None, lines=()):
    """
    Renders vertical bar group chart

    :param legend - list of legend values.
        Example: ['bar1', 'bar2', 'bar3']
    :param dataset - list of values for each type (value, deviation)
        Example:
            [
                [(10,1), (11, 2), (10,1)], # bar1 values
                [(30,(29,33)),(35,(33,36)), (30,(29,33))], # bar2 values
                [(20,(19,21)),(20,(13, 24)), (20,(19,21))] # bar 3 values
            ]
    :param width - width of chart
    :param height - height of chart
    :param scale_x - x ace scale
    :param scale_y - y ace scale

    :returns url to chart

    dataset example:
    {
        'relese_1': {
            'randr': (1, 0.1),
            'randwr': (2, 0.2)
        }
        'release_2': {
            'randr': (3, 0.3),
            'randwr': (4, 0.4)
        }
    }
    """

    bar = VerticalBarGroup([], encoding='text')
    bar.title(title)

    dataset = bars_data + bars_dev_top + bars_dev_bottom + \
        [l[0] for l in lines]

    bar.dataset(dataset, series=len(bars_data))
    bar.axes.type('xyy')
    bar.axes.label(2, None, label_x)

    if scale_x:
        bar.axes.label(0, *scale_x)

    max_value = (max([max(l) for l in dataset[:2]]))
    bar.axes.range(1, 0, max_value)
    bar.axes.style(1, 'N*s*')
    bar.axes.style(2, '000000', '13')

    bar.scale(*[0, max_value] * 3)

    bar.bar('r', '.1', '1')
    for i in range(1):
        bar.marker('E', '000000', '%s:%s' % ((len(bars_data) + i*2), i),
                   '', '1:10')
    bar.color(*COLORS)
    bar.size(width, height)

    axes_type = "xyy"

    scale = [0, max_value] * len(bars_dev_top + bars_dev_bottom + bars_data)
    if lines:
        line_n = 0
        for data, label, axe, leg in lines:
            bar.marker('D', COLORS[len(bars_data) + line_n],
                       (len(bars_data + bars_dev_top + bars_dev_bottom))
                       + line_n, 0, 3)
            # max_val_l = max(data)
            if axe:
                max_val_l = max(data)
                bar.axes.type(axes_type + axe)
                bar.axes.range(len(axes_type), 0, max_val_l)
                bar.axes.style(len(axes_type), 'N*s*')
                bar.axes.label(len(axes_type) + 1, None, label)
                bar.axes.style(len(axes_type) + 1, '000000', '13')
                axes_type += axe
                line_n += 1
                scale += [0, max_val_l]
            else:
                scale += [0, max_value]
            legend.append(leg)
            # scale += [0, max_val_l]

    bar.legend(*legend)
    bar.scale(*scale)
    img_name = file_name + ".png"
    img_path = os.path.join(cfg_dict['charts_img_path'], img_name)

    if not os.path.exists(img_path):
        bar.save(img_path)

    return get_top_top_dir(img_path)


def render_lines(title, legend, dataset, scale_x, width=700, height=400):
    line = Line([], encoding="text")
    line.title(title)
    line.dataset(dataset)

    line.axes('xy')
    max_value = (max([max(l) for l in dataset]))
    line.axes.range(1, 0, max_value)
    line.scale(0, max_value)
    line.axes.label(0, *scale_x)
    line.legend(*legend)
    line.color(*COLORS[:len(legend)])
    line.size(width, height)

    img_name = hashlib.md5(str(line)).hexdigest() + ".png"
    img_path = os.path.join(cfg_dict['charts_img_path'], img_name)
    if not os.path.exists(img_path):
        line.save(img_path)

    return get_top_top_dir(img_path)
