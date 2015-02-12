from GChartWrapper import VerticalBarGroup
from GChartWrapper import Line
from GChartWrapper import constants


COLORS = ["1569C7", "81D8D0", "307D7E", "5CB3FF", "blue", "indigo"]
constants.MARKERS += 'E'  # append E marker to available markers


def render_vertical_bar(title, legend, dataset, width=700, height=400, scale_x=None,
                        scale_y=None):
    """
    Renders vertical bar group chart

    :param legend - list of legend values.
        Example: ['bar1', 'bar2', 'bar3']
    :param dataset - list of values for each type (value, deviation)
        Example:
            [
                [(10,(9,11)), (11, (3,12)), (10,(9,11))], # bar1 values
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

    values = []
    deviations = []

    for d in dataset:
        val, dev = zip(*d)

        display_dev = []
        for i in range(len(val)):
            display_dev.append((val[i]-dev[i], val[i]+dev[i]))
        values.append(val)
        # deviations.extend(zip(*dev))
        deviations.extend(zip(*display_dev))

    bar.dataset(values + deviations, series=len(values))
    bar.axes.type('xyy')
    bar.axes.label(2, None, 'kbps')
    if scale_x:
        bar.axes.label(0, *scale_x)

    max_value = (max([max(l) for l in values + deviations]))
    bar.axes.range(1, 0, max_value)
    bar.axes.style(1, 'N*s*')
    bar.axes.style(2, '000000', '13')

    bar.scale(0, max_value)

    bar.bar('r', '.1', '1')
    for i in range(len(legend)):
        bar.marker('E', '000000', '%s:%s' % ((len(values) + i*2), i),
                   '', '1:10')
    bar.legend(*legend)
    bar.color(*COLORS[:len(values)])
    bar.size(width, height)

    return bar


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
    return str(line)
