# ----------------  PROFILES  ------------------------------------------------------------------------------------------


# this is default values, real values is loaded from config
class ColorProfile:
    primary_color = 'b'
    suppl_color1 = 'teal'
    suppl_color2 = 'magenta'
    suppl_color3 = 'orange'
    box_color = 'y'
    err_color = 'red'

    noise_alpha = 0.3
    subinfo_alpha = 0.7

    imshow_colormap = None  # type: str
    hmap_cmap = "Blues"


default_format = 'svg'
io_chart_format = 'svg'


class StyleProfile:
    default_style = 'seaborn-white'
    io_chart_style = 'classic'

    dpi = 80

    lat_samples = 5

    tide_layout = False
    hist_boxes = 10
    hist_lat_boxes = 25
    hm_hist_bins_count = 25
    hm_x_slots = 25
    min_points_for_dev = 5

    x_label_rotation = 35

    dev_range_x = 2.0
    dev_perc = 95

    point_shape = 'o'
    err_point_shape = '*'

    avg_range = 20
    approx_average = True
    approx_average_no_points = False

    curve_approx_level = 6
    curve_approx_points = 100
    assert avg_range >= min_points_for_dev

    # figure size in inches
    figsize = (8, 4)
    figsize_long = (8, 4)
    qd_chart_inches = (16, 9)

    subplot_adjust_r = 0.75
    subplot_adjust_r_no_legend = 0.9
    title_font_size = 12

    extra_io_spine = True

    legend_for_eng = True

    # heatmap interpolation is deprecated
    # heatmap_interpolation = '1d'
    # heatmap_interpolation = None
    # heatmap_interpolation_points = 300

    heatmap_colorbar = False
    outliers_q_nd = 3.0
    outliers_hide_q_nd = 4.0
    outliers_lat = (0.01, 0.9)

    violin_instead_of_box = True
    violin_point_count = 30000

    min_iops_vs_qd_jobs = 3

    qd_bins = [0, 1, 2, 4, 6, 8, 12, 16, 20, 26, 32, 40, 48, 56, 64, 96, 128]
    iotime_bins = list(range(0, 1030, 50))
    block_size_bins = [0, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048]
    large_blocks = 256

    min_load_diff = 0.05

    histo_grid = 'x'


DefColorProfile = ColorProfile()
DefStyleProfile = StyleProfile()
