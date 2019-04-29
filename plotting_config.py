import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', size=10)
rc('lines', linewidth=1)
# rc('lines', markersize=1)
# rc('scatter', marker='|')
rc('axes', grid=True)
rc('patch', edgecolor='black')
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

cmap = plt.get_cmap('winter')
cm = lambda x: cmap(1 - x)

pdf_params = {
    'format': 'pdf',
    'bbox_inches': 'tight',
    # 'dpi': 1000
}
png_params = {
    'format': 'png',
    'bbox_inches': 'tight',
    # 'dpi': 1000
}
svg_params = {
    'format': 'svg',
    'bbox_inches': 'tight',
    # 'dpi': 1000
}

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


half_width_square = (3.15, 3.15)
full_width_square = (6.3, 6.3)

def savefig(fig_name, pdf=True, svg=True, png=True):
    if png: plt.savefig(fig_name + '.png', **png_params)
    if svg: plt.savefig(fig_name + '.svg', **svg_params)
    if pdf: plt.savefig(fig_name + '.pdf', **pdf_params)