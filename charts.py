from GChartWrapper import VerticalBarGroup
from GChartWrapper import Line
from GChartWrapper import constants


COLORS = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]
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

    bar.axes.type('xy')
    if scale_x:
        bar.axes.label(0, *scale_x)
    scale_y = scale_y or range(int(max([max(l) for l in values]) + 2))
    bar.axes.range(1, *scale_y)

    bar.bar('r', '.1', '1')
    for i in range(len(legend)):
        bar.marker('E', '000000', '%s:%s' % ((len(values) + i*2), i),
                   '', '1:10')
    bar.legend(*legend)
    bar.color(*COLORS[:len(values)])
    bar.size(width, height)

    return str(bar)


def render_lines():
    line = Line([])
    line.dataset([[1,2,3], [3,2,1], [5,6,7]])
    scale_y = range(int(max([max(l) for l in
                             [[1,2,3], [3,2,1], [5,6,7]]]) + 2))
    line.axes.range(1, *scale_y)
    # G.legend('Animals','Vegetables','Minerals')
    # G.axes('y')
