"""Exploratory data analysis charts rendered as pure SVG.

Every chart on the EDA page is generated here as inline SVG markup, so the
dashboard needs no charting library, no JavaScript and no network access.

Charts provided:

    line       trend over a sequence (hour of day, time)
    bar        category averages (month, weekday)
    histogram  distribution of a single numeric column
    box        five number summary per group
    scatter    relationship between two numeric columns
    heatmap    correlation matrix
    donut      share of a whole

"""

import html
import math

import numpy as np
import pandas as pd

from load_data import (
    DATE_COLUMN,
    DEFAULT_REGION,
    TARGET,
    get_region_label,
    load_data,
    resolve_region,
)
from preprocessing import clean_series


# =========================================================
# DESIGN TOKENS
# =========================================================

# One palette for the whole page so the charts read as a single design
# (Apple system colours, light theme)
PALETTE = [
    "#0A84FF",  # blue
    "#5E5CE6",  # indigo
    "#BF5AF2",  # purple
    "#FF375F",  # pink
    "#FF9F0A",  # orange
    "#FFD60A",  # yellow
    "#30D158",  # green
    "#64D2FF",  # periwinkle
    "#FF6482",  # rose
    "#32ADE6",  # sky
]

INK = "#1d1d1f"

MUTED = "#6e6e73"

GRID = "rgba(60, 60, 67, 0.12)"

AXIS = "rgba(60, 60, 67, 0.28)"

CHART_W = 780

CHART_H = 340

MARGIN_LEFT = 80

MARGIN_RIGHT = 26

MARGIN_TOP = 26

MARGIN_BOTTOM = 64

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

WEEKDAY_LABELS = [
    "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
]


# =========================================================
# SMALL HELPERS
# =========================================================

def _esc(value) -> str:
    """Escape anything that ends up inside an SVG attribute or text node."""

    return html.escape(str(value), quote=True)


def _n(value) -> str:
    """Coordinate formatted to two decimals, keeping the file readable."""

    return f"{float(value):.2f}"


def _fmt(value) -> str:
    """Compact axis tick label."""

    value = float(value)

    magnitude = abs(value)

    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if magnitude >= 10_000:
        return f"{value / 1_000:.0f}k"

    if magnitude >= 1_000:
        return f"{value / 1_000:.1f}k"

    if magnitude >= 10:
        return f"{value:.0f}"

    if magnitude == 0:
        return "0"

    if magnitude >= 1:
        return f"{value:.1f}"

    return f"{value:.2f}"


def _ticks(low: float, high: float, count: int = 5) -> list:
    """Round numbers spread across the domain."""

    if not math.isfinite(low) or not math.isfinite(high):
        return []

    if high <= low:
        high = low + 1.0

    span = high - low

    raw = span / max(count, 1)

    if raw <= 0:
        return [low]

    magnitude = 10 ** math.floor(math.log10(raw))

    normalised = raw / magnitude

    if normalised <= 1:
        step = 1.0 * magnitude
    elif normalised <= 2:
        step = 2.0 * magnitude
    elif normalised <= 2.5:
        step = 2.5 * magnitude
    elif normalised <= 5:
        step = 5.0 * magnitude
    else:
        step = 10.0 * magnitude

    value = math.ceil(low / step) * step

    result = []

    guard = 0

    while value <= high + step * 1e-9 and guard < 64:

        result.append(round(value, 10))

        value += step

        guard += 1

    return result or [low, high]


def _mapper(low: float, high: float, start: float, end: float):
    """Map a value domain onto pixels."""

    span = (high - low) or 1.0

    def map_value(value):

        return start + (float(value) - low) / span * (end - start)

    return map_value


def _svg_open(width: int, height: int, label: str) -> str:

    return (
        f'<svg class="chart-svg" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_esc(label)}" '
        f'preserveAspectRatio="xMidYMid meet">'
        f'<title>{_esc(label)}</title>'
    )


def _empty(label: str) -> str:
    """Placeholder shown when a chart has nothing to draw."""

    return (
        _svg_open(CHART_W, 160, label)
        + f'<rect x="0" y="0" width="{CHART_W}" height="160" '
          'fill="rgba(60,60,67,0.04)" rx="14"/>'
        + f'<text x="{CHART_W / 2}" y="86" text-anchor="middle" '
          f'fill="{MUTED}" font-size="15">No data available</text>'
        + '</svg>'
    )


def _y_grid(map_y, left: float, width: float, ticks: list) -> str:
    """Horizontal grid lines with tick labels on the left."""

    parts = []

    for tick in ticks:

        y = map_y(tick)

        if y < MARGIN_TOP - 2 or y > MARGIN_TOP + width + 2:
            continue

        parts.append(
            f'<line x1="{_n(left)}" y1="{_n(y)}" '
            f'x2="{_n(left + width)}" y2="{_n(y)}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )

        parts.append(
            f'<text x="{_n(left - 12)}" y="{_n(y + 4)}" '
            f'text-anchor="end" fill="{MUTED}" font-size="11">'
            f'{_esc(_fmt(tick))}</text>'
        )

    return "".join(parts)


def _axis_titles(width: int, height: int, xlabel: str, ylabel: str) -> str:
    """Optional axis captions drawn inside the plot frame."""

    parts = []

    if ylabel:

        parts.append(
            f'<text x="16" y="{_n((MARGIN_TOP + height - MARGIN_BOTTOM) / 2)}" '
            f'fill="{MUTED}" font-size="11" text-anchor="middle" '
            f'transform="rotate(-90 16 '
            f'{_n((MARGIN_TOP + height - MARGIN_BOTTOM) / 2)})">'
            f'{_esc(ylabel)}</text>'
        )

    if xlabel:

        parts.append(
            f'<text x="{_n(width / 2)}" y="{height - 8}" '
            f'fill="{MUTED}" font-size="11" text-anchor="middle">'
            f'{_esc(xlabel)}</text>'
        )

    return "".join(parts)


def _label_step(count: int, capacity: int) -> int:
    """Thin out x labels so they never collide."""

    if count <= capacity:
        return 1

    return int(math.ceil(count / capacity))


# =========================================================
# LINE / AREA CHART
# =========================================================

def line_chart(labels, values, label, *, color=PALETTE[0],
               area=True, xlabel="", ylabel="", dots=True) -> str:
    """Trend line across an ordered sequence."""

    values = [float(v) for v in values]

    labels = [str(v) for v in labels]

    if not values:

        return _empty(label)

    count = len(values)

    low, high = min(values), max(values)

    if high == low:

        high = low + 1.0

    span = high - low

    # Keep a little headroom so the curve never touches the frame
    low -= span * 0.10
    high += span * 0.10

    if min(values) >= 0 and low < 0:

        low = 0.0

    plot_w = CHART_W - MARGIN_LEFT - MARGIN_RIGHT

    plot_h = CHART_H - MARGIN_TOP - MARGIN_BOTTOM

    map_x = _mapper(0, max(count - 1, 1), MARGIN_LEFT, MARGIN_LEFT + plot_w)

    map_y = _mapper(low, high, MARGIN_TOP + plot_h, MARGIN_TOP)

    parts = [_svg_open(CHART_W, CHART_H, label)]

    defs_id = f"g{abs(hash(label)) % 100000}"

    parts.append(
        '<defs>'
        f'<linearGradient id="{defs_id}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity="0.42"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0.02"/>'
        f'</linearGradient>'
        f'<linearGradient id="{defs_id}s" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity="0.55"/>'
        f'<stop offset="100%" stop-color="{color}"/>'
        f'</linearGradient>'
        '</defs>'
    )

    parts.append(_y_grid(map_y, MARGIN_LEFT, plot_h, _ticks(low, high)))

    points = [(map_x(i), map_y(value)) for i, value in enumerate(values)]

    line_path = " ".join(
        f"{'M' if i == 0 else 'L'}{_n(x)} {_n(y)}"
        for i, (x, y) in enumerate(points)
    )

    if area:

        baseline = MARGIN_TOP + plot_h

        area_path = (
            f"M{_n(points[0][0])} {_n(baseline)} "
            + " ".join(f"L{_n(x)} {_n(y)}" for x, y in points)
            + f" L{_n(points[-1][0])} {_n(baseline)} Z"
        )

        parts.append(
            f'<path d="{area_path}" fill="url(#{defs_id})"/>'
        )

    parts.append(
        f'<path d="{line_path}" fill="none" '
        f'stroke="url(#{defs_id}s)" stroke-width="2.6" '
        'stroke-linejoin="round" stroke-linecap="round"/>'
    )

    if dots and count <= 40:

        for x, y in points:

            parts.append(
                f'<circle cx="{_n(x)}" cy="{_n(y)}" r="3.4" '
                f'fill="#ffffff" stroke="{color}" stroke-width="2.2"/>'
            )

    step = _label_step(count, 14)

    parts.append(
        f'<line x1="{_n(MARGIN_LEFT)}" y1="{_n(MARGIN_TOP + plot_h)}" '
        f'x2="{_n(MARGIN_LEFT + plot_w)}" y2="{_n(MARGIN_TOP + plot_h)}" '
        f'stroke="{AXIS}" stroke-width="1.2"/>'
    )

    for index, text in enumerate(labels):

        if index % step:
            continue

        parts.append(
            f'<line x1="{_n(map_x(index))}" y1="{_n(MARGIN_TOP + plot_h)}" '
            f'x2="{_n(map_x(index))}" y2="{_n(MARGIN_TOP + plot_h + 6)}" '
            f'stroke="{AXIS}" stroke-width="1"/>'
        )

        parts.append(
            f'<text x="{_n(map_x(index))}" y="{_n(MARGIN_TOP + plot_h + 24)}" '
            f'text-anchor="middle" fill="{MUTED}" font-size="11">'
            f'{_esc(text)}</text>'
        )

    parts.append(_axis_titles(CHART_W, CHART_H, xlabel, ylabel))

    parts.append('</svg>')

    return "".join(parts)


# =========================================================
# BAR CHART
# =========================================================

def bar_chart(labels, values, label, *, color=PALETTE[1],
              xlabel="", ylabel="", colors=None) -> str:
    """Vertical bars for category averages.

    ``colors`` optionally gives each bar its own shade; when omitted every
    bar shares the ``color`` gradient.
    """

    values = [float(v) for v in values]

    labels = [str(v) for v in labels]

    if not values:

        return _empty(label)

    count = len(values)

    low, high = min(values), max(values)

    # Bars always start at zero unless the data actually goes negative
    low = min(low, 0.0)

    if high == low:

        high = low + 1.0

    span = high - low

    high += span * 0.12

    plot_w = CHART_W - MARGIN_LEFT - MARGIN_RIGHT

    plot_h = CHART_H - MARGIN_TOP - MARGIN_BOTTOM

    map_y = _mapper(low, high, MARGIN_TOP + plot_h, MARGIN_TOP)

    parts = [_svg_open(CHART_W, CHART_H, label)]

    defs_id = f"b{abs(hash(label)) % 100000}"

    parts.append(
        '<defs>'
        f'<linearGradient id="{defs_id}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity="0.98"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0.34"/>'
        f'</linearGradient>'
        '</defs>'
    )

    parts.append(_y_grid(map_y, MARGIN_LEFT, plot_h, _ticks(low, high)))

    slot = plot_w / count

    bar_w = max(6.0, slot * 0.62)

    for index, value in enumerate(values):

        centre = MARGIN_LEFT + slot * (index + 0.5)

        y = map_y(value)

        height = max(1.5, (MARGIN_TOP + plot_h) - y)

        if colors:

            fill = colors[index % len(colors)]

        else:

            fill = f"url(#{defs_id})"

        parts.append(
            f'<rect x="{_n(centre - bar_w / 2)}" y="{_n(y)}" '
            f'width="{_n(bar_w)}" height="{_n(height)}" rx="5" '
            f'fill="{fill}"/>'
        )

        if count <= 16:

            parts.append(
                f'<text x="{_n(centre)}" y="{_n(y - 8)}" '
                f'text-anchor="middle" fill="{INK}" font-size="11" '
                f'font-weight="600">{_esc(_fmt(value))}</text>'
            )

    step = _label_step(count, 14)

    parts.append(
        f'<line x1="{_n(MARGIN_LEFT)}" y1="{_n(MARGIN_TOP + plot_h)}" '
        f'x2="{_n(MARGIN_LEFT + plot_w)}" y2="{_n(MARGIN_TOP + plot_h)}" '
        f'stroke="{AXIS}" stroke-width="1.2"/>'
    )

    for index, text in enumerate(labels):

        if index % step:
            continue

        centre = MARGIN_LEFT + slot * (index + 0.5)

        parts.append(
            f'<text x="{_n(centre)}" y="{_n(MARGIN_TOP + plot_h + 24)}" '
            f'text-anchor="middle" fill="{MUTED}" font-size="11">'
            f'{_esc(text)}</text>'
        )

    parts.append(_axis_titles(CHART_W, CHART_H, xlabel, ylabel))

    parts.append('</svg>')

    return "".join(parts)


# =========================================================
# HISTOGRAM
# =========================================================

def histogram(counts, edges, label, *, color=PALETTE[3]) -> str:
    """Distribution of one numeric column across equal width bins."""

    counts = [float(c) for c in counts]

    edges = [float(e) for e in edges]

    if not counts or len(edges) < 2:

        return _empty(label)

    low, high = 0.0, max(counts)

    if high <= 0:

        return _empty(label)

    high *= 1.14

    plot_w = CHART_W - MARGIN_LEFT - MARGIN_RIGHT

    plot_h = CHART_H - MARGIN_TOP - MARGIN_BOTTOM

    map_y = _mapper(low, high, MARGIN_TOP + plot_h, MARGIN_TOP)

    parts = [_svg_open(CHART_W, CHART_H, label)]

    defs_id = f"h{abs(hash(label)) % 100000}"

    parts.append(
        '<defs>'
        f'<linearGradient id="{defs_id}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity="0.95"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0.30"/>'
        f'</linearGradient>'
        '</defs>'
    )

    parts.append(_y_grid(map_y, MARGIN_LEFT, plot_h, _ticks(low, high)))

    span = edges[-1] - edges[0]

    def map_edge(value):

        return MARGIN_LEFT + (value - edges[0]) / span * plot_w

    for index, count in enumerate(counts):

        x0 = map_edge(edges[index])
        x1 = map_edge(edges[index + 1])

        y = map_y(count)

        height = max(1.5, (MARGIN_TOP + plot_h) - y)

        # One pixel of breathing room keeps the bars readable
        parts.append(
            f'<rect x="{_n(x0 + 1)}" y="{_n(y)}" '
            f'width="{_n(max(1.0, x1 - x0 - 2))}" height="{_n(height)}" '
            f'rx="3" fill="url(#{defs_id})"/>'
        )

    step = _label_step(len(counts), 12)

    parts.append(
        f'<line x1="{_n(MARGIN_LEFT)}" y1="{_n(MARGIN_TOP + plot_h)}" '
        f'x2="{_n(MARGIN_LEFT + plot_w)}" y2="{_n(MARGIN_TOP + plot_h)}" '
        f'stroke="{AXIS}" stroke-width="1.2"/>'
    )

    for index in range(0, len(edges), step):

        x = map_edge(edges[index])

        parts.append(
            f'<line x1="{_n(x)}" y1="{_n(MARGIN_TOP + plot_h)}" '
            f'x2="{_n(x)}" y2="{_n(MARGIN_TOP + plot_h + 6)}" '
            f'stroke="{AXIS}" stroke-width="1"/>'
        )

        parts.append(
            f'<text x="{_n(x)}" y="{_n(MARGIN_TOP + plot_h + 24)}" '
            f'text-anchor="middle" fill="{MUTED}" font-size="11">'
            f'{_esc(_fmt(edges[index]))}</text>'
        )

    parts.append(_axis_titles(CHART_W, CHART_H, "Demand (MW)", "Frequency"))

    parts.append('</svg>')

    return "".join(parts)


# =========================================================
# BOX PLOT
# =========================================================

def box_plot(groups, label, *, color=PALETTE[5], ylabel="") -> str:
    """Five number summary per group with outlier dots."""

    groups = [g for g in groups if g.get("q1") is not None]

    if not groups:

        return _empty(label)

    all_values = []

    for group in groups:

        all_values += [
            group["whisker_low"],
            group["whisker_high"],
            group["q1"],
            group["q3"],
        ]

        all_values += list(group.get("outliers", []))

    low, high = min(all_values), max(all_values)

    if high == low:

        high = low + 1.0

    span = high - low

    low -= span * 0.07
    high += span * 0.07

    plot_w = CHART_W - MARGIN_LEFT - MARGIN_RIGHT

    plot_h = CHART_H - MARGIN_TOP - MARGIN_BOTTOM

    map_y = _mapper(low, high, MARGIN_TOP + plot_h, MARGIN_TOP)

    parts = [_svg_open(CHART_W, CHART_H, label)]

    parts.append(_y_grid(map_y, MARGIN_LEFT, plot_h, _ticks(low, high)))

    count = len(groups)

    slot = plot_w / count

    box_w = max(10.0, slot * 0.52)

    for index, group in enumerate(groups):

        centre = MARGIN_LEFT + slot * (index + 0.5)

        shade = PALETTE[index % len(PALETTE)]

        q1, med, q3 = group["q1"], group["med"], group["q3"]

        w_lo, w_hi = group["whisker_low"], group["whisker_high"]

        # Whisker stems
        parts.append(
            f'<line x1="{_n(centre)}" y1="{_n(map_y(w_hi))}" '
            f'x2="{_n(centre)}" y2="{_n(map_y(q3))}" '
            f'stroke="{shade}" stroke-width="1.6"/>'
        )

        parts.append(
            f'<line x1="{_n(centre)}" y1="{_n(map_y(q1))}" '
            f'x2="{_n(centre)}" y2="{_n(map_y(w_lo))}" '
            f'stroke="{shade}" stroke-width="1.6"/>'
        )

        # Whisker caps
        for value in (w_hi, w_lo):

            parts.append(
                f'<line x1="{_n(centre - box_w * 0.34)}" '
                f'y1="{_n(map_y(value))}" '
                f'x2="{_n(centre + box_w * 0.34)}" '
                f'y2="{_n(map_y(value))}" '
                f'stroke="{shade}" stroke-width="2"/>'
            )

        # Box
        box_top = map_y(q3)
        box_bottom = map_y(q1)

        parts.append(
            f'<rect x="{_n(centre - box_w / 2)}" y="{_n(box_top)}" '
            f'width="{_n(box_w)}" '
            f'height="{_n(max(2.0, box_bottom - box_top))}" rx="4" '
            f'fill="{shade}" fill-opacity="0.26" '
            f'stroke="{shade}" stroke-width="1.6"/>'
        )

        # Median
        parts.append(
            f'<line x1="{_n(centre - box_w / 2)}" y1="{_n(map_y(med))}" '
            f'x2="{_n(centre + box_w / 2)}" y2="{_n(map_y(med))}" '
            f'stroke="{shade}" stroke-width="2.6" stroke-linecap="round"/>'
        )

        # Outliers, sampled so dense groups stay fast to draw
        outliers = list(group.get("outliers", []))

        if len(outliers) > 40:

            pick = np.linspace(0, len(outliers) - 1, 40).astype(int)

            outliers = [outliers[i] for i in pick]

        for value in outliers:

            parts.append(
                f'<circle cx="{_n(centre)}" cy="{_n(map_y(value))}" '
                f'r="2.4" fill="{shade}" fill-opacity="0.75"/>'
            )

    parts.append(
        f'<line x1="{_n(MARGIN_LEFT)}" y1="{_n(MARGIN_TOP + plot_h)}" '
        f'x2="{_n(MARGIN_LEFT + plot_w)}" y2="{_n(MARGIN_TOP + plot_h)}" '
        f'stroke="{AXIS}" stroke-width="1.2"/>'
    )

    for index, group in enumerate(groups):

        centre = MARGIN_LEFT + slot * (index + 0.5)

        parts.append(
            f'<text x="{_n(centre)}" y="{_n(MARGIN_TOP + plot_h + 24)}" '
            f'text-anchor="middle" fill="{MUTED}" font-size="11">'
            f'{_esc(group["label"])}</text>'
        )

    parts.append(_axis_titles(CHART_W, CHART_H, "Month", ylabel))

    parts.append('</svg>')

    return "".join(parts)


# =========================================================
# SCATTER
# =========================================================

def scatter(xs, ys, label, *, color=PALETTE[2],
            xlabel="", ylabel="", trend=True) -> str:
    """Relationship between two numeric columns with a fitted trend line."""

    xs = np.asarray(xs, dtype=float)

    ys = np.asarray(ys, dtype=float)

    if xs.size == 0:

        return _empty(label)

    x_low, x_high = float(np.min(xs)), float(np.max(xs))

    y_low, y_high = float(np.min(ys)), float(np.max(ys))

    if x_high == x_low:
        x_high = x_low + 1.0

    if y_high == y_low:
        y_high = y_low + 1.0

    x_pad = (x_high - x_low) * 0.06
    y_pad = (y_high - y_low) * 0.08

    x_low -= x_pad
    x_high += x_pad
    y_low -= y_pad
    y_high += y_pad

    if y_low < 0 and float(np.min(ys)) >= 0:
        y_low = 0.0

    plot_w = CHART_W - MARGIN_LEFT - MARGIN_RIGHT

    plot_h = CHART_H - MARGIN_TOP - MARGIN_BOTTOM

    map_x = _mapper(x_low, x_high, MARGIN_LEFT, MARGIN_LEFT + plot_w)

    map_y = _mapper(y_low, y_high, MARGIN_TOP + plot_h, MARGIN_TOP)

    parts = [_svg_open(CHART_W, CHART_H, label)]

    parts.append(_y_grid(map_y, MARGIN_LEFT, plot_h, _ticks(y_low, y_high)))

    # Points, drawn first so the trend line sits above them
    plot_x = []

    for index in range(0, xs.size, 1):

        px = map_x(xs[index])
        py = map_y(ys[index])

        if py < MARGIN_TOP - 4 or py > MARGIN_TOP + plot_h + 4:
            continue

        plot_x.append((px, py, index))

        parts.append(
            f'<circle cx="{_n(px)}" cy="{_n(py)}" r="3" '
            f'fill="{color}" fill-opacity="0.42" '
            f'stroke="{color}" stroke-opacity="0.55" stroke-width="0.8"/>'
        )

    correlation = 0.0

    if xs.size >= 2 and float(np.std(xs)) > 0 and float(np.std(ys)) > 0:

        slope, intercept = np.polyfit(xs, ys, 1)

        correlation = float(np.corrcoef(xs, ys)[0, 1])

        if trend:

            x0, x1 = x_low, x_high

            parts.append(
                f'<line x1="{_n(map_x(x0))}" y1="{_n(map_y(slope * x0 + intercept))}" '
                f'x2="{_n(map_x(x1))}" y2="{_n(map_y(slope * x1 + intercept))}" '
                f'stroke="{color}" stroke-width="2.4" stroke-dasharray="7 5" '
                'stroke-linecap="round"/>'
            )

    # x tick labels
    x_ticks = _ticks(x_low, x_high, 6)

    parts.append(
        f'<line x1="{_n(MARGIN_LEFT)}" y1="{_n(MARGIN_TOP + plot_h)}" '
        f'x2="{_n(MARGIN_LEFT + plot_w)}" y2="{_n(MARGIN_TOP + plot_h)}" '
        f'stroke="{AXIS}" stroke-width="1.2"/>'
    )

    for tick in x_ticks:

        x = map_x(tick)

        if x < MARGIN_LEFT - 1 or x > MARGIN_LEFT + plot_w + 1:
            continue

        parts.append(
            f'<line x1="{_n(x)}" y1="{_n(MARGIN_TOP + plot_h)}" '
            f'x2="{_n(x)}" y2="{_n(MARGIN_TOP + plot_h + 6)}" '
            f'stroke="{AXIS}" stroke-width="1"/>'
        )

        parts.append(
            f'<text x="{_n(x)}" y="{_n(MARGIN_TOP + plot_h + 24)}" '
            f'text-anchor="middle" fill="{MUTED}" font-size="11">'
            f'{_esc(_fmt(tick))}</text>'
        )

    # Correlation badge in the top right corner
    parts.append(
        f'<rect x="{MARGIN_LEFT + plot_w - 108}" y="{MARGIN_TOP + 6}" '
        'width="102" height="30" rx="15" '
        'fill="rgba(255,255,255,0.85)" stroke="rgba(60,60,67,0.18)"/>'
        f'<text x="{MARGIN_LEFT + plot_w - 57}" y="{MARGIN_TOP + 26}" '
        f'text-anchor="middle" fill="{INK}" font-size="12" '
        f'font-weight="600">r = {correlation:.3f}</text>'
    )

    parts.append(_axis_titles(CHART_W, CHART_H, xlabel, ylabel))

    parts.append('</svg>')

    return "".join(parts)


# =========================================================
# CORRELATION HEATMAP
# =========================================================

def _mix(first, second, amount: float) -> str:

    def channel(index):

        value = first[index] + (second[index] - first[index]) * amount

        return max(0, min(255, int(round(value))))

    return "#{:02x}{:02x}{:02x}".format(channel(0), channel(1), channel(2))


def _diverging(value: float) -> str:

    base = (238, 240, 246)

    if value >= 0:

        target = (10, 132, 255)

        amount = min(1.0, abs(value))

    else:

        target = (255, 55, 95)

        amount = min(1.0, abs(value))

    return _mix(base, target, amount)


def heatmap(matrix, row_labels, col_labels, label) -> str:
    """Diverging correlation matrix."""

    if not matrix or not matrix[0]:

        return _empty(label)

    rows = len(matrix)

    cols = len(matrix[0])

    left = 148

    top = 74

    right = 34

    bottom = 30

    cell_w = (CHART_W - left - right) / cols

    cell_h = 46

    height = top + rows * cell_h + bottom

    parts = [_svg_open(CHART_W, height, label)]

    for index, text in enumerate(col_labels):

        centre = left + cell_w * (index + 0.5)

        parts.append(
            f'<text x="{_n(centre)}" y="{top - 14}" '
            f'text-anchor="start" fill="{MUTED}" font-size="11" '
            f'transform="rotate(-38 {_n(centre)} {top - 14})">'
            f'{_esc(text)}</text>'
        )

    for row in range(rows):

        centre_y = top + cell_h * row + cell_h / 2

        parts.append(
            f'<text x="{left - 14}" y="{_n(centre_y + 4)}" '
            f'text-anchor="end" fill="{INK}" font-size="11.5">'
            f'{_esc(row_labels[row])}</text>'
        )

        for col in range(cols):

            value = float(matrix[row][col])

            x = left + cell_w * col

            y = top + cell_h * row

            fill = _diverging(value)

            strong = abs(value) > 0.55

            parts.append(
                f'<rect x="{_n(x + 2)}" y="{_n(y + 2)}" '
                f'width="{_n(cell_w - 4)}" height="{_n(cell_h - 4)}" '
                f'rx="7" fill="{fill}"/>'
            )

            parts.append(
                f'<text x="{_n(x + cell_w / 2)}" y="{_n(y + cell_h / 2 + 4)}" '
                f'text-anchor="middle" font-size="12" '
                f'font-weight="{"700" if strong else "500"}" '
                f'fill="{"#ffffff" if strong else "#1d1d1f"}">'
                f'{value:+.2f}</text>'
            )

    parts.append(
        f'<text x="{left}" y="{height - 8}" fill="{MUTED}" font-size="11">'
        'Pearson correlation, −1 (rose) through 0 to +1 (blue)</text>'
    )

    parts.append('</svg>')

    return "".join(parts)


# =========================================================
# DONUT
# =========================================================

def _point(cx, cy, radius, degrees):

    radians = math.radians(degrees)

    return cx + radius * math.cos(radians), cy + radius * math.sin(radians)


def donut_chart(labels, values, label, *, colors=None) -> str:
    """Share of a whole rendered as a ring with a legend."""

    values = [float(v) for v in values]

    labels = [str(v) for v in labels]

    total = sum(values)

    if not values or total <= 0:

        return _empty(label)

    colors = colors or PALETTE

    width, height = CHART_W, CHART_H

    cx, cy = 196, height / 2

    outer, inner = 126, 74

    parts = [_svg_open(width, height, label)]

    start = -90.0

    running = 0.0

    for index, value in enumerate(values):

        share = value / total

        sweep = share * 360.0

        end = start + sweep

        colour = colors[index % len(colors)]

        if sweep >= 359.99:

            # A single slice is a full ring, which needs two arcs
            parts.append(
                f'<circle cx="{_n(cx)}" cy="{_n(cy)}" r="{_n((outer + inner) / 2)}" '
                f'fill="none" stroke="{colour}" '
                f'stroke-width="{_n(outer - inner)}"/>'
            )

        elif sweep > 0.01:

            x0, y0 = _point(cx, cy, outer, start)
            x1, y1 = _point(cx, cy, outer, end)
            x2, y2 = _point(cx, cy, inner, end)
            x3, y3 = _point(cx, cy, inner, start)

            large = 1 if sweep > 180 else 0

            parts.append(
                f'<path d="M{_n(x0)} {_n(y0)} '
                f'A{_n(outer)} {_n(outer)} 0 {large} 1 {_n(x1)} {_n(y1)} '
                f'L{_n(x2)} {_n(y2)} '
                f'A{_n(inner)} {_n(inner)} 0 {large} 0 {_n(x3)} {_n(y3)} Z" '
                f'fill="{colour}" fill-opacity="0.88" '
                'stroke="#ffffff" stroke-width="1.5"/>'
            )

        running += share

        start = end

    parts.append(
        f'<text x="{_n(cx)}" y="{_n(cy - 6)}" text-anchor="middle" '
        f'fill="{INK}" font-size="30" font-weight="700">'
        f'{_fmt(total)}</text>'
        f'<text x="{_n(cx)}" y="{_n(cy + 18)}" text-anchor="middle" '
        f'fill="{MUTED}" font-size="11.5">total MW</text>'
    )

    # Legend
    legend_x = 388

    legend_y = 44

    step_y = min(30.0, (height - legend_y - 24) / max(len(values), 1))

    for index, (name, value) in enumerate(zip(labels, values)):

        colour = colors[index % len(colors)]

        share = 100.0 * value / total

        y = legend_y + step_y * index

        parts.append(
            f'<rect x="{legend_x}" y="{_n(y - 9)}" width="14" height="14" '
            f'rx="4" fill="{colour}" fill-opacity="0.9"/>'
            f'<text x="{legend_x + 24}" y="{_n(y + 2)}" fill="{INK}" '
            f'font-size="12">{_esc(name)}</text>'
            f'<text x="{width - 34}" y="{_n(y + 2)}" text-anchor="end" '
            f'fill="{MUTED}" font-size="12">'
            f'{_fmt(value)} &middot; {share:.1f}%</text>'
        )

    parts.append('</svg>')

    return "".join(parts)


# =========================================================
# MULTI LINE OVERLAY
# =========================================================

def multiline_chart(series, labels, xlabels, label, *, colors=None) -> str:
    """Several trend lines on one frame, coloured per series."""

    series = [[float(v) for v in row] for row in series]

    labels = [str(v) for v in labels]

    xlabels = [str(v) for v in xlabels]

    if not series or not series[0]:

        return _empty(label)

    count = max(len(row) for row in series)

    all_values = [v for row in series for v in row if math.isfinite(v)]

    if not all_values:

        return _empty(label)

    low, high = min(all_values), max(all_values)

    if high == low:

        high = low + 1.0

    span = high - low

    low -= span * 0.10
    high += span * 0.10

    if min(all_values) >= 0 and low < 0:

        low = 0.0

    plot_w = CHART_W - MARGIN_LEFT - MARGIN_RIGHT

    plot_h = CHART_H - MARGIN_TOP - MARGIN_BOTTOM

    map_x = _mapper(0, max(count - 1, 1), MARGIN_LEFT, MARGIN_LEFT + plot_w)

    map_y = _mapper(low, high, MARGIN_TOP + plot_h, MARGIN_TOP)

    cols = list(colors) if colors else PALETTE

    parts = [_svg_open(CHART_W, CHART_H, label)]

    parts.append(_y_grid(map_y, MARGIN_LEFT, plot_h, _ticks(low, high)))

    for index, row in enumerate(series):

        colour = cols[index % len(cols)]

        path = []

        for point, value in enumerate(row):

            if not math.isfinite(value):
                continue

            path.append(
                f"{'M' if not path else 'L'}"
                f"{_n(map_x(point))} {_n(map_y(value))}"
            )

        if path:

            parts.append(
                f'<path d="{" ".join(path)}" fill="none" '
                f'stroke="{colour}" stroke-width="1.7" '
                'stroke-linejoin="round" stroke-linecap="round" '
                f'stroke-opacity="0.62"/>'
            )

    # x grid and labels
    x_ticks = []

    for point in range(count):

        x_ticks.append(point)

    step = _label_step(count, 13)

    parts.append(
        f'<line x1="{_n(MARGIN_LEFT)}" y1="{_n(MARGIN_TOP + plot_h)}" '
        f'x2="{_n(MARGIN_LEFT + plot_w)}" y2="{_n(MARGIN_TOP + plot_h)}" '
        f'stroke="{AXIS}" stroke-width="1.2"/>'
    )

    for point in range(0, count, step):

        x = map_x(point)

        parts.append(
            f'<line x1="{_n(x)}" y1="{_n(MARGIN_TOP + plot_h)}" '
            f'x2="{_n(x)}" y2="{_n(MARGIN_TOP + plot_h + 6)}" '
            f'stroke="{AXIS}" stroke-width="1"/>'
        )

        parts.append(
            f'<text x="{_n(x)}" y="{_n(MARGIN_TOP + plot_h + 24)}" '
            f'text-anchor="middle" fill="{MUTED}" font-size="11">'
            f'{_esc(xlabels[point])}</text>'
        )

    # Compact legend in the top right corner
    legend_x = MARGIN_LEFT + plot_w - 168

    legend_y = MARGIN_TOP + 12

    rows_per_column = 6

    columns = math.ceil(len(labels) / rows_per_column)

    legend_w = columns * 88

    legend_h = min(len(labels), rows_per_column) * 17 + 16

    parts.append(
        f'<rect x="{_n(legend_x - 8)}" y="{_n(legend_y - 6)}" '
        f'width="{_n(legend_w)}" height="{_n(legend_h)}" rx="12" '
        'fill="rgba(255,255,255,0.88)" stroke="rgba(60,60,67,0.16)"/>'
    )

    for index, name in enumerate(labels):

        column = index // rows_per_column

        row = index % rows_per_column

        colour = cols[index % len(cols)]

        x = legend_x + column * 88

        y = legend_y + row * 17

        parts.append(
            f'<line x1="{_n(x)}" y1="{_n(y)}" x2="{_n(x + 16)}" '
            f'y2="{_n(y)}" stroke="{colour}" stroke-width="3" '
            f'stroke-opacity="0.85" stroke-linecap="round"/>'
        )

        parts.append(
            f'<text x="{_n(x + 22)}" y="{_n(y + 4)}" fill="{MUTED}" '
            f'font-size="10.5">{_esc(name)}</text>'
        )

    parts.append(_axis_titles(CHART_W, CHART_H, "Hour of day", "Demand (MW)"))

    parts.append('</svg>')

    return "".join(parts)


# =========================================================
# SEQUENTIAL VALUE HEATMAP
# =========================================================

def _sequential(value: float, low: float, high: float) -> str:

    base = (236, 246, 255)

    target = (0, 113, 227)

    amount = 0.0 if high <= low else (value - low) / (high - low)

    amount = max(0.0, min(1.0, amount))

    return _mix(base, target, amount)


def value_heatmap(matrix, row_labels, col_labels, label, *, unit="") -> str:
    """Colour grid for magnitude data, unlike the diverging correlation map."""

    if not matrix or not matrix[0]:

        return _empty(label)

    rows = len(matrix)

    cols = len(matrix[0])

    left = 148

    top = 74

    right = 34

    bottom = 30

    cell_w = (CHART_W - left - right) / cols

    cell_h = 46

    height = top + rows * cell_h + bottom

    flat = [float(v) for row in matrix for v in row if v is not None]

    low_value = min(flat) if flat else 0.0

    high_value = max(flat) if flat else 1.0

    parts = [_svg_open(CHART_W, height, label)]

    for index, text in enumerate(col_labels):

        centre = left + cell_w * (index + 0.5)

        parts.append(
            f'<text x="{_n(centre)}" y="{top - 14}" '
            f'text-anchor="start" fill="{MUTED}" font-size="11" '
            f'transform="rotate(-38 {_n(centre)} {top - 14})">'
            f'{_esc(text)}</text>'
        )

    show_text = cell_w >= 34 and rows <= 16

    for row in range(rows):

        centre_y = top + cell_h * row + cell_h / 2

        parts.append(
            f'<text x="{left - 14}" y="{_n(centre_y + 4)}" '
            f'text-anchor="end" fill="{INK}" font-size="11.5">'
            f'{_esc(row_labels[row])}</text>'
        )

        for col in range(cols):

            raw = matrix[row][col]

            if raw is None:

                continue

            value = float(raw)

            x = left + cell_w * col

            y = top + cell_h * row

            fill = _sequential(value, low_value, high_value)

            parts.append(
                f'<rect x="{_n(x + 2)}" y="{_n(y + 2)}" '
                f'width="{_n(cell_w - 4)}" height="{_n(cell_h - 4)}" '
                f'rx="7" fill="{fill}"/>'
            )

            if show_text:

                strong = (value - low_value) / (high_value - low_value) > 0.55

                parts.append(
                    f'<text x="{_n(x + cell_w / 2)}" '
                    f'y="{_n(y + cell_h / 2 + 4)}" '
                    f'text-anchor="middle" font-size="11.5" '
                    f'font-weight="{"700" if strong else "400"}" '
                    f'fill="{"#ffffff" if strong else "#1d1d1f"}">'
                    f'{_esc(_fmt(value))}</text>'
                )

    parts.append(
        f'<text x="{left}" y="{height - 8}" fill="{MUTED}" font-size="11">'
        f'colour scale {_fmt(low_value)} → {_fmt(high_value)} '
        f'({_esc(unit or "value")})</text>'
    )

    parts.append('</svg>')

    return "".join(parts)


# =========================================================
# REPORT
# =========================================================

def _bucket_by_days(dated: pd.Series, maximum: int = 140) -> tuple:
    """Collapse a daily series into a readable number of labels."""

    labels = [str(d) for d in dated.index]

    series = np.asarray(dated.to_numpy(), dtype=float)

    if series.size <= maximum:

        return labels, [float(v) for v in series]

    buckets = min(maximum, 100)

    pieces = np.array_split(np.arange(series.size), buckets)

    grouped_labels = []

    grouped_values = []

    for piece in pieces:

        if piece.size == 0:
            continue

        grouped_labels.append(labels[int(piece[0])][5:])

        grouped_values.append(float(np.mean(series[piece])))

    return grouped_labels, grouped_values


def get_eda_report(region: str = DEFAULT_REGION) -> dict:
    """Every EDA chart for one region, cached per region."""

    region = resolve_region(region)

    cache_key = f"eda::{region}"

    from load_data import _CACHE  # reuse the loader cache

    if cache_key in _CACHE:

        return _CACHE[cache_key]

    cleaned, _ = clean_series(load_data(region))

    timestamps = cleaned[DATE_COLUMN]

    demand = cleaned[TARGET]

    # -----------------------------------------------------
    # Descriptive statistics
    # -----------------------------------------------------

    stats = {
        "rows": int(len(cleaned)),
        "mean": round(float(demand.mean()), 2),
        "median": round(float(demand.median()), 2),
        "std": round(float(demand.std()), 2),
        "min": round(float(demand.min()), 2),
        "max": round(float(demand.max()), 2),
        "skewness": round(float(demand.skew()), 3),
        "kurtosis": round(float(demand.kurtosis()), 3),
        "iqr": round(
            float(demand.quantile(0.75) - demand.quantile(0.25)), 2
        ),
        "start": str(timestamps.min()),
        "end": str(timestamps.max()),
    }

    charts = []

    # -----------------------------------------------------
    # 1. Line - mean demand by hour of day
    # -----------------------------------------------------

    hourly = demand.groupby(timestamps.dt.hour).mean()

    charts.append({
        "key": "hourly",
        "title": "Mean Demand by Hour of Day",
        "subtitle": "Line chart of the average load across all 24 hours",
        "kind": "Line chart",
        "wide": False,
        "svg": line_chart(
            [f"{int(h):02d}:00" for h in hourly.index],
            hourly.to_numpy(),
            "Mean demand by hour of day",
            color=PALETTE[0],
            xlabel="Hour of day",
            ylabel="Demand (MW)",
        ),
    })

    # -----------------------------------------------------
    # 2. Line - daily trend over time
    # -----------------------------------------------------

    daily = (
        cleaned
        .assign(**{DATE_COLUMN: timestamps.dt.normalize()})
        .groupby(DATE_COLUMN)[TARGET]
        .mean()
    )

    labels, values = _bucket_by_days(daily)

    charts.append({
        "key": "trend",
        "title": "Demand Trend Over Time",
        "subtitle": "Line chart of average daily demand across the archive",
        "kind": "Line chart",
        "wide": True,
        "svg": line_chart(
            labels,
            values,
            "Demand trend over time",
            color=PALETTE[1],
            area=True,
            dots=len(values) <= 40,
            xlabel="Date",
            ylabel="Demand (MW)",
        ),
    })

    # -----------------------------------------------------
    # 3. Bar - mean demand by month
    # -----------------------------------------------------

    monthly = demand.groupby(timestamps.dt.month).mean()

    charts.append({
        "key": "monthly",
        "title": "Mean Demand by Month",
        "subtitle": "Bar chart comparing the twelve calendar months",
        "kind": "Bar chart",
        "wide": False,
        "svg": bar_chart(
            [MONTH_LABELS[int(m) - 1] for m in monthly.index],
            monthly.to_numpy(),
            "Mean demand by month",
            color=PALETTE[3],
            xlabel="Month",
            ylabel="Demand (MW)",
        ),
    })

    # -----------------------------------------------------
    # 4. Bar - mean demand by weekday
    # -----------------------------------------------------

    weekday = demand.groupby(timestamps.dt.dayofweek).mean()

    charts.append({
        "key": "weekday",
        "title": "Mean Demand by Weekday",
        "subtitle": "Bar chart showing the weekday to weekend drop",
        "kind": "Bar chart",
        "wide": False,
        "svg": bar_chart(
            [WEEKDAY_LABELS[int(d)] for d in weekday.index],
            weekday.to_numpy(),
            "Mean demand by weekday",
            color=PALETTE[4],
            xlabel="Day of week",
            ylabel="Demand (MW)",
        ),
    })

    # -----------------------------------------------------
    # 5. Histogram - distribution of demand
    # -----------------------------------------------------

    counts, edges = np.histogram(demand.to_numpy(), bins=32)

    charts.append({
        "key": "histogram",
        "title": "Distribution of Demand",
        "subtitle": "Histogram of hourly load readings in equal width bins",
        "kind": "Histogram",
        "wide": False,
        "svg": histogram(
            counts.tolist(),
            edges.tolist(),
            "Distribution of demand",
            color=PALETTE[3],
        ),
    })

    # -----------------------------------------------------
    # 6. Box plot - spread of demand per month
    # -----------------------------------------------------

    groups = []

    for month, values in demand.groupby(timestamps.dt.month):

        q1 = float(values.quantile(0.25))
        med = float(values.quantile(0.50))
        q3 = float(values.quantile(0.75))

        spread = q3 - q1

        low_fence = q1 - 1.5 * spread
        high_fence = q3 + 1.5 * spread

        inside = values[(values >= low_fence) & (values <= high_fence)]

        outliers = values[(values < low_fence) | (values > high_fence)]

        groups.append({
            "label": MONTH_LABELS[int(month) - 1],
            "q1": q1,
            "med": med,
            "q3": q3,
            "whisker_low": float(inside.min()),
            "whisker_high": float(inside.max()),
            "outliers": [float(v) for v in outliers.to_numpy()],
        })

    charts.append({
        "key": "boxplot",
        "title": "Demand Spread by Month",
        "subtitle": "Box plot of quartiles, whiskers and outliers per month",
        "kind": "Box plot",
        "wide": False,
        "svg": box_plot(
            groups,
            "Demand spread by month",
            color=PALETTE[5],
            ylabel="Demand (MW)",
        ),
    })

    # -----------------------------------------------------
    # 7. Scatter - this hour against the same hour yesterday
    # -----------------------------------------------------

    lagged = demand.shift(24)

    valid = lagged.notna() & demand.notna()

    scatter_x = lagged[valid].to_numpy()

    scatter_y = demand[valid].to_numpy()

    if scatter_x.size > 700:

        pick = np.linspace(0, scatter_x.size - 1, 700).astype(int)

        scatter_x = scatter_x[pick]

        scatter_y = scatter_y[pick]

    charts.append({
        "key": "scatter",
        "title": "Load Against Load 24 Hours Earlier",
        "subtitle": "Scatter plot with a fitted trend line and correlation",
        "kind": "Scatter plot",
        "wide": True,
        "svg": scatter(
            scatter_x,
            scatter_y,
            "Load against load 24 hours earlier",
            color=PALETTE[2],
            xlabel="Demand 24 hours ago (MW)",
            ylabel="Demand now (MW)",
            trend=True,
        ),
    })

    # -----------------------------------------------------
    # 8. Correlation heatmap
    # -----------------------------------------------------

    frame = pd.DataFrame({
        "hour": timestamps.dt.hour,
        "weekday": timestamps.dt.dayofweek,
        "month": timestamps.dt.month,
        "is_weekend": (timestamps.dt.dayofweek >= 5).astype(int),
        "is_peak_hour": (
            ((timestamps.dt.hour >= 7) & (timestamps.dt.hour <= 9)
             & (timestamps.dt.dayofweek < 5))
            | ((timestamps.dt.hour >= 17) & (timestamps.dt.hour <= 20)
               & (timestamps.dt.dayofweek < 5))
        ).astype(int),
        "lag_1h": demand.shift(1),
        "lag_24h": demand.shift(24),
        "roll_24h": demand.shift(1).rolling(24, min_periods=1).mean(),
        "Demand_MW": demand,
    })

    correlation = frame.corr()

    charts.append({
        "key": "heatmap",
        "title": "Correlation Matrix",
        "subtitle": "Heatmap of Pearson correlation between the EDA variables",
        "kind": "Heatmap",
        "wide": True,
        "svg": heatmap(
            correlation.to_numpy().round(2).tolist(),
            list(correlation.index),
            list(correlation.columns),
            "Correlation matrix",
        ),
    })

    # -----------------------------------------------------
    # 9. Donut - share of total demand by month
    # -----------------------------------------------------

    monthly_total = demand.groupby(timestamps.dt.month).sum()

    charts.append({
        "key": "donut",
        "title": "Share of Annual Demand by Month",
        "subtitle": "Donut chart of the proportion of total load per month",
        "kind": "Pie chart",
        "wide": False,
        "svg": donut_chart(
            [MONTH_LABELS[int(m) - 1] for m in monthly_total.index],
            monthly_total.to_numpy(),
            "Share of annual demand by month",
            colors=PALETTE,
        ),
    })

    report = {
        "region": region,
        "region_label": get_region_label(region),
        "file": f"{region}.csv",
        "value_column": TARGET,
        "stats": stats,
        "charts": charts,
        "chart_count": len(charts),
    }

    _CACHE[cache_key] = report

    return report


if __name__ == "__main__":

    import json

    import sys

    wanted = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REGION

    result = get_eda_report(wanted)

    stripped = dict(result)

    stripped["charts"] = [
        {k: (f"<{len(v)} chars of svg>" if k == "svg" else v)
         for k, v in chart.items()}
        for chart in result["charts"]
    ]

    print(json.dumps(stripped, indent=2, default=str))
