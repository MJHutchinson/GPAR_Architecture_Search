import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', size=10)
rc('lines', linewidth=0.5)
# rc('lines', markersize=1)
# rc('scatter', marker='|')
rc('axes', grid=True)
rc('patch', edgecolor='black')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

cmap = plt.get_cmap('winter')
cm = lambda x: cmap(1 - x)

fig_params = {
    'format': 'pdf',
    'bbox_inches': 'tight',
    'dpi': 1000
}
png_params = {
    'format': 'png',
    'bbox_inches': 'tight',
    'dpi': 1000
}

half_width_square = (3.15, 3.15)
full_width_square = (6.3, 6.3)