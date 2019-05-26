import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', size=10)
rc('lines', linewidth=0.5)
rc('lines', markersize=6)
rc('scatter', marker='+')
rc('axes', grid=True)
rc('axes', axisbelow=True)
rc('patch', edgecolor='black')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('legend', fontsize=7)

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
linestyles = ['-','--','-.']


half_width_square = (2.8, 2.8)
full_width_square = (5.6, 5.6)

text_width = 5.8
text_height = 9.1

def savefig(fig_name, pdf=False, svg=False, png=True):
    if png: plt.savefig(fig_name + '.png', **png_params)
    if svg: plt.savefig(fig_name + '.svg', **svg_params)
    if pdf: plt.savefig(fig_name + '.pdf', **pdf_params)

def savefig_handle(fig, fig_name, pdf=False, svg=False, png=True):
    if png: fig.savefig(fig_name + '.png', **png_params)
    if svg: fig.savefig(fig_name + '.svg', **svg_params)
    if pdf: fig.savefig(fig_name + '.pdf', **pdf_params)