from typing import List, Union
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams.update({'font.size': 17})
plt.rc('legend', **{'fontsize': 12})
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['grid.linewidth'] = 1
mpl.rcParams['lines.linewidth'] = 1


def add_title(ax, title, font_size, left_position, top_position):
    """
    add the title of the plot
    """
    ax.text(left_position, top_position, title,
            fontsize=font_size,
            fontweight='bold',
            horizontalalignment='left',
            verticalalignment='center')


def add_label_y_axis(ax, title, font_size, left_position, top_position):
    """
    add label of the y axis at a given position.
    """
    ax.text(left_position, top_position, title, fontsize=font_size,
            style='italic',
            horizontalalignment='left',
            verticalalignment='center')


def plot_confidence_interval(ax, df_comb_eff, experiment_names, marker_size_scale, summary_row_position=1.0):
    """
    plotting of the confidence interval of the dataset including data rows and summary rows
    """
    # plot confident interval of the data
    for index, name in enumerate(experiment_names, 2):
        i = index + summary_row_position
        ax.plot(
            (df_comb_eff.ci_low[name], df_comb_eff.ci_upp[name]),
            ([i, i]),
            color='darkgrey',
            linewidth=1
        )
        ax.plot(
            (df_comb_eff.ci_low[name], df_comb_eff.ci_low[name]),
            ([i-0.1, i+0.1]),
            color='darkgrey',
            linewidth=1,
            clip_on=False
        )
        ax.plot(
            (df_comb_eff.ci_upp[name], df_comb_eff.ci_upp[name]),
            ([i-0.1, i+0.1]),
            color='darkgrey',
            linewidth=1,
            clip_on=False
        )
        # marker
        marker_size = marker_size_scale * (1 + df_comb_eff.w_re[name])\
            if marker_size_scale is not None else mpl.rcParams['lines.linewidth'] * 2
        ax.plot(
            df_comb_eff.eff[name],
            i,
            marker='D',
            markersize=marker_size,
            color='black'
        )

    # plot confident interval of the summary
    ax.plot(
        (df_comb_eff.ci_low['combined_effect'],
         df_comb_eff.ci_upp['combined_effect']),
        (summary_row_position, summary_row_position),
        color='darkgrey',
        linewidth=1
    )
    ax.plot(
        (df_comb_eff.ci_low['combined_effect'],
         df_comb_eff.ci_low['combined_effect']),
        (summary_row_position - 0.1, summary_row_position + 0.1),
        color='darkgrey',
        linewidth=1,
        clip_on=False
    )
    ax.plot(
        (df_comb_eff.ci_upp['combined_effect'],
         df_comb_eff.ci_upp['combined_effect']),
        (summary_row_position - 0.1, summary_row_position + 0.1),
        color='darkgrey',
        linewidth=1,
        clip_on=False
    )
    marker_size = marker_size_scale * 2\
        if marker_size_scale is not None else mpl.rcParams['lines.linewidth'] * 2
    ax.plot(
        df_comb_eff.eff['combined_effect'],
        summary_row_position,
        marker='D',
        markersize=marker_size,
        color='black'
    )


def determine_max_text_width(fig, ax, items):
    """
    Return maximum text width in data coordinate system
    """
    inv = ax.transData.inverted()
    item_widths = []
    item_xmin = inv.transform((0, 0))[0]
    renderer = fig.canvas.get_renderer()
    for item in items:
        tmp_text = plt.text(0, 0, item)
        item_widths.append(inv.transform((tmp_text.get_window_extent(renderer=renderer).width, 0))[0] - item_xmin)
        tmp_text.remove()
    max_width = max(item_widths)
    return max_width


def add_column(ax, items, header, summary, offset, text_alignment, summary_row_position, header_offset, color):
    """
    add a column which include header, data rows and a summary.
    """
    # summary row starts at the first row.
    texts = [ax.text(
        offset,
        summary_row_position,
        summary,
        horizontalalignment=text_alignment,
        verticalalignment='center'
    )]
    # data rows start at 2 rows from the summary.
    # the row under is meant to be a line separating data and summary section.
    for i, item in enumerate(items, 2):
        if item is None:
            continue
        texts.append(ax.text(
            offset,
            i + summary_row_position,
            item,
            horizontalalignment=text_alignment,
            verticalalignment='center',
            color=color
        ))
    # the header row starts at the next row after the data row.
    if header is not None and len(header) > 0:
        texts.append(ax.text(
            offset,
            len(items) + summary_row_position + header_offset + 1,
            header,
            horizontalalignment=text_alignment,
            verticalalignment='center',
            style='italic',
        ))
    return texts


def add_columns(
        ax,
        columns,
        column_offset,
        column_spacing,
        column_widths,
        column_headers,
        column_summary,
        text_alignment,
        is_negative_offset=False,
        summary_row_position=1.0,
        header_offset=1.0,
        colors=None
):
    """
    add columns one by one starting from x = `column offset` (data coordinate) and move to the next column by shifting by
    column spacing and column width.
    """
    if colors is None:
        colors = ['k' for _ in columns]
    texts = []
    offset = column_offset + column_spacing
    for i, column in enumerate(columns):
        texts += add_column(
            ax, column, column_headers[i], column_summary[i],
            offset, text_alignment, summary_row_position,
            header_offset, colors[i]
        )
        if is_negative_offset:
            offset += column_spacing - column_widths[i]
        else:
            offset += column_spacing + column_widths[i]
    return texts


def init(ax, df_comb_eff, experiment_names, label_x_axis, zero_line):
    ax.set_ylim((0, len(experiment_names) + 2))
    xmin = df_comb_eff.ci_low.min()
    xmax = df_comb_eff.ci_upp.max()
    if zero_line:
        # to ensure that the x-axis contain 0
        xmin = min(0, xmin)
        xmax = max(0, xmax)
    ax.set_xlim((xmin, xmax))
    ax.set_yticks(range(0, len(experiment_names) + 5))
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(label_x_axis)


def determine_column_spacing(ax, fontsize):
    """
    determine the spacing between each column the size returned is approximately the size 1 character. as it is
    calculated with fontsize (in pixel) and transformed into data coordinate
    """
    inv = ax.transData.inverted()
    width_reference = inv.transform((0, 0))[0]
    space_in_data_coord= inv.transform((fontsize, 0))[0]
    return space_in_data_coord - width_reference


def add_zero_line(ax, ymax):
    """
    vertical line at x=0 in the plot.
    """
    ax.axvline(
        x=0,
        # transform data coordinate into axes coordinate (range from 0 to 1)
        ymax=ax.transAxes.inverted().transform(ax.transData.transform((0, ymax)))[1],
        linestyle=':',
        color='k'
    )


def add_line(fig, ax, from_pos, to_pos):
    inv = fig.transFigure.inverted()
    # transform data coordinate into figure coordinate (range from 0 to 1)
    from_pos_fig_coord = inv.transform(ax.transData.transform(from_pos))
    to_pos_fig_coord = inv.transform(ax.transData.transform(to_pos))
    line = mpl.lines.Line2D(
        (from_pos_fig_coord[0], to_pos_fig_coord[0]),
        (from_pos_fig_coord[1], to_pos_fig_coord[1]),
        color='k'
    )
    fig.add_artist(line)


def calculate_column_spacing(
        fig,
        ax,
        font_size,
        left_columns,
        left_column_headers,
        left_column_summary,
        right_columns,
        right_column_headers,
        right_column_summary
):
    """
    This function prepares data for each columns including column format, column headers, and summary.
    """
    # just for determining column width
    tmp_left_columns = [
        left_columns[i] + [left_column_headers[i]] + [left_column_summary[i]]
        for i in range(len(left_columns))
    ]
    # use negative for shifting left
    left_column_widths = [
        determine_max_text_width(fig, ax, column) for column in tmp_left_columns
    ]
    # just for determining column width
    tmp_right_columns = [
        right_columns[i] + [right_column_headers[i]] + [right_column_summary[i]]
        for i in range(len(right_columns))
    ]
    right_column_widths = [
        determine_max_text_width(fig, ax, column) for column in tmp_right_columns
    ]
    column_spacing = determine_column_spacing(ax, font_size)
    return (
        left_column_widths,
        right_column_widths,
        column_spacing
    )


def calculate_layout(
        fig,
        ax,
        fig_width,
        fig_height,
        column_spacing,
        row_spacing,
        num_left_column,
        left_column_widths,
        num_right_columns,
        right_column_widths,
    ):
    """
    This function calculates the width of the left and the right side of the main plot in the data coordinate.
    Then, transforms them into figure coordinate and find out how much space is needed for each side.
    The outputs are the offset margin for the left and the right size of the plot.
    """
    total_left = sum(left_column_widths)
    total_right = sum(right_column_widths)
    xmin = ax.transData.transform((0, 0))[0]
    left = ax.transData.transform((total_left, 0))[0] + column_spacing * len(left_column_widths) - xmin
    right = ax.transData.transform((total_right, 0))[0] + column_spacing * len(right_column_widths) - xmin

    renderer = fig.canvas.get_renderer()
    # Use 'M' as a reference character.
    tmp_text = plt.text(0, 0, 'M')
    text_height = tmp_text.get_window_extent(renderer=renderer).height
    tmp_text.remove()

    # 2 rows for 'summary' and 'x-axis'
    bottom_margin = 2
    # plus 1 for the summary row
    num_content_rows = max(num_left_column, num_right_columns) + 1
    # times with 2 for spacing. The unit is in pixel because font size is in pixel.
    bottom = bottom_margin * text_height * (1 + row_spacing)
    top = bottom + text_height * num_content_rows * (1 + row_spacing)
    figure_width = fig_width * fig.dpi
    figure_height = fig_height * fig.dpi
    return left / figure_width, 1 - right / figure_width, bottom / figure_height, top / figure_height


def prepare_plot_info(ax):
    ### SPACING/LOCATION
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    label_y_axis = 'Effect Size'
    return (
        xmin, xmax,
        ymin, ymax,
        label_y_axis
    )


def prepare_plot_data(df_comb_eff, experiment_names):
    df_comb_eff = df_comb_eff
    ### LEFT COLUMN
    left_column_headers = [""]
    left_column_summary = ['Summary Effect (RE)']
    left_columns = [experiment_names]
    ### RIGHT COLUMN
    right_column_headers = ['Weight', 'Mean', 'CI 95%']
    summary_mask = df_comb_eff.index.isin(['combined_effect'])
    right_column_summary = [
        '',
        df_comb_eff[summary_mask]['eff'].map('{:.2f}'.format)[0],
        (
                '[' + '{:.2f}'.format(df_comb_eff[summary_mask]['ci_low'][0])
                + ', ' + '{:.2f}'.format(df_comb_eff[summary_mask]['ci_upp'][0]) + ']'
        ),
    ]
    experiment_mask = df_comb_eff.index.isin(experiment_names)
    ci_column = (
            '[' + df_comb_eff[experiment_mask]['ci_low'].map('{:.2f}'.format)
            + ', ' + df_comb_eff[experiment_mask]['ci_upp'].map('{:.2f}'.format) + ']'
    )
    right_columns = [
        (df_comb_eff[experiment_mask]['w_re'] * 100).map('{:.1f}%'.format).tolist(),
        df_comb_eff[experiment_mask]['eff'].map('{:.2f}'.format).tolist(),
        ci_column.tolist(),
    ]
    return (
        left_columns,
        left_column_headers,
        left_column_summary,
        right_columns,
        right_column_headers,
        right_column_summary,
    )

def calculate_row_positions(num_experiments, summary_row_position=1.0, header_offset=1.0):
    title_position = num_experiments + summary_row_position + header_offset + 2
    title_line_position = num_experiments + summary_row_position + header_offset + 1.5
    label_y_axis_position = num_experiments + summary_row_position + header_offset + 1
    header_line_position = num_experiments + summary_row_position + header_offset + 0.5
    summary_line_position = summary_row_position + 1
    zero_line_top_position = header_line_position
    return (
        title_position,
        title_line_position,
        label_y_axis_position,
        header_line_position,
        summary_line_position,
        zero_line_top_position
    )


def check_dataframe(df_comb_eff, experiment_names):
    columns = {'ci_low', 'ci_upp', 'w_re', 'eff'}
    if not {'ci_low', 'ci_upp', 'w_re', 'eff'}.issubset(df_comb_eff.columns):
        raise TypeError(f'Input dataframe `df_comb_eff` must contain the following columns {list(columns)}')

    if not set(experiment_names + ['combined_effect']).issubset(df_comb_eff.index):
        raise TypeError(f'Input dataframe `df_comb_eff` must contain all given rows/experiments {experiment_names}')


def forest_plot(
        effect_size:pd.DataFrame,
        experiment_names:List[str],
        title:str,
        label_x_axis:str,

        add_left_columns:Union[None,List[List[str]]]=None,
        add_left_column_headers:Union[None,List[str]]=None,
        add_left_column_summary:Union[None,List[str]]=None,
        
        add_right_columns:Union[None,List[List[str]]]=None,
        add_right_column_headers:Union[None,List[str]]=None,
        add_right_column_summary:Union[None,List[str]]=None,
        
        fig_width=6,
        fig_height=4,
        zero_line=True,
        marker_size_scale=8,
        font_size=13,
        row_spacing=2,
):
    """
    Plot a forest plot.

    General Idea
    ------------

    The layout is determined separately in the vertical position and the horizontal position, but this function uses
    data coordinate for both axis.

    The vertical position is determined by the y-axis which is simply row counts (see `init()`).

    The horizontal position is determined by the width of the maximum item of each columns with a spacing of
    about 1 character.

    Parameters
    ----------
    left_columns: List[List[str]]
    left_column_headers: List[str]
    left_column_summary: List[str]
    right_columns: List[List[str]]
    right_column_headers: List[str]
    right_column_summary: List[str]
    title: str
    experiment_names: List[str]
    label_x_axis: str
    effect_size : pd.DataFrame
        A dataframe of effect size analysis with `ci_low`, `ci_upp`, `eff`, `w_re` columns and `combined_effect` row
    fig_width: int
    fig_height: int
    zero_line: bool
        Enable the dashed line at 0 or not
    marker_size_scale: int
        Size of the marker on the confidence interval. Disable by assigning None
    font_size: int
    row_spacing: int
        Size of vertical spacing between rows. 1 unit is approximately the height of a character.
    plot_file: str
        The output PDF filepath of the plot
    """
    check_dataframe(effect_size, experiment_names)
    # Left and right margin in term of figure coordinate (range from 0 to 1)
    PAGE_MARGIN = 0.05
    # Position of the summary row
    SUMMARY_ROW_POSITION = 0.5
    # offset position of the header
    HEADER_OFFSET = 1.5

    (
        left_columns,
        left_column_headers,
        left_column_summary,
        right_columns,
        right_column_headers,
        right_column_summary,
    ) = prepare_plot_data(effect_size, experiment_names)

    if add_left_columns is not None and add_left_column_headers is not None and add_left_column_summary is not None:
        left_columns.extend(add_left_columns)
        left_column_headers.extend(add_left_column_headers)
        left_column_summary.extend(add_left_column_summary)

    if add_right_columns is not None and add_right_column_headers is not None and add_right_column_summary is not None:
        right_columns.extend(add_right_columns)
        right_column_headers.extend(add_right_column_headers)
        right_column_summary.extend(add_right_column_summary)


    experiment_names = [] if experiment_names is None else experiment_names
    num_experiments = len(experiment_names)
    # reverse the list because we plot things from the bottom up (from index 0)
    # so it is more convenience to reverse it once instead of manipulating with the index later.
    experiment_names = list(reversed(experiment_names))
    left_columns = list(reversed([list(reversed(column)) for column in left_columns]))
    left_column_headers = list(reversed(left_column_headers))
    left_column_summary = list(reversed(left_column_summary))
    right_columns = [list(reversed(column)) for column in right_columns]

    effect_size = effect_size[::-1]

    # style forest plot
    mpl.rcParams['font.size'] = font_size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    (
        left_column_widths,
        right_column_widths,
        column_spacing
    ) = calculate_column_spacing(
        fig,
        ax,
        font_size,
        left_columns,
        left_column_headers,
        left_column_summary,
        right_columns,
        right_column_headers,
        right_column_summary
    )
    left, right, bottom, top = calculate_layout(
        fig,
        ax,
        fig_width,
        fig_height,
        column_spacing,
        row_spacing,
        len(left_columns[0]),
        left_column_widths,
        len(right_columns[0]),
        right_column_widths,
    )

    # here we adjust the axes area (the main plotting area) to have enough space for the left
    # and right columns.
    plt.subplots_adjust(left=left + PAGE_MARGIN, right=right - PAGE_MARGIN, bottom=bottom, top=top)

    ### This section prepare the data for the plot and relevant layout config
    init(ax, effect_size, experiment_names, label_x_axis, zero_line)

    # calculate column spacing again because we just changed the axes area above
    (
        left_column_widths,
        right_column_widths,
        column_spacing
    ) = calculate_column_spacing(
        fig,
        ax,
        font_size,
        left_columns,
        left_column_headers,
        left_column_summary,
        right_columns,
        right_column_headers,
        right_column_summary
    )
    (
        xmin, xmax,
        ymin, ymax,
        label_y_axis
    ) = prepare_plot_info(ax)

    (
        title_position,
        title_line_position,
        label_y_axis_position,
        header_line_position,
        summary_line_position,
        zero_line_top_position
    ) = calculate_row_positions(
        num_experiments,
        summary_row_position=SUMMARY_ROW_POSITION,
        header_offset=HEADER_OFFSET
    )
    ###

    add_title(ax, title, font_size, xmin, title_position)
    # line between the title and the headers
    add_line(
        fig,
        ax,
        from_pos=(xmin, title_line_position),
        to_pos=(xmax + sum(right_column_widths)+ column_spacing * len(right_column_widths), title_line_position)
    )
    add_label_y_axis(ax, label_y_axis, font_size, xmin, label_y_axis_position)

    # add columns on the right side
    add_columns(
        ax, right_columns, xmax, column_spacing,
        right_column_widths, right_column_headers,
        right_column_summary, text_alignment='left',
        summary_row_position=SUMMARY_ROW_POSITION,
        header_offset=HEADER_OFFSET,
        colors=['#777777', 'k', 'k', 'k', 'k', 'k', 'k']
    )
    # add columns on the left side
    add_columns(
        ax, left_columns, xmin, -column_spacing,
        left_column_widths, left_column_headers,
        left_column_summary, text_alignment='right',
        is_negative_offset=True,
        summary_row_position=SUMMARY_ROW_POSITION,
        header_offset=HEADER_OFFSET
    )
    plot_confidence_interval(
        ax, effect_size, experiment_names, marker_size_scale, summary_row_position=SUMMARY_ROW_POSITION
    )

    # line between the headers and the data rows
    add_line(
        fig,
        ax,
        from_pos=(
            xmin - sum(left_column_widths) - column_spacing * len(left_column_widths),
            header_line_position
        ),
        to_pos=(
            xmax + sum(right_column_widths) + column_spacing * len(right_column_widths),
            header_line_position
        )
    )
    # the line between the summary row and the data rows
    add_line(
        fig,
        ax,
        from_pos=(
            xmin - sum(left_column_widths) - column_spacing * len(left_column_widths),
            summary_line_position
        ),
        to_pos=(
            xmax + sum(right_column_widths) + column_spacing * len(right_column_widths),
            summary_line_position
        )
    )

    if zero_line:
        add_zero_line(ax, ymax=zero_line_top_position)

    return fig