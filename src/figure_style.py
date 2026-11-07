"""Warm paper, terracotta and muted botanical colors for report figures."""
PAPER = '#faf9f5'
INK = '#3d3934'
TERRACOTTA = '#c96442'
OLIVE = '#6f7b5c'
SAND = '#b79263'

def apply_style(plt, font):
    plt.rcParams.update({
        'font.family': font, 'font.size': 10, 'axes.unicode_minus': False,
        'svg.fonttype': 'none', 'figure.facecolor': PAPER,
        'axes.facecolor': PAPER, 'savefig.facecolor': PAPER,
        'text.color': INK, 'axes.labelcolor': INK,
        'axes.edgecolor': '#b5ada1', 'xtick.color': INK, 'ytick.color': INK,
        'grid.color': '#d8d1c5', 'grid.alpha': .6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.facecolor': PAPER, 'legend.edgecolor': '#ddd6ca',
        'legend.framealpha': .95, 'lines.linewidth': 1.7,
    })
