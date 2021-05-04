import matplotlib
import matplotlib.pyplot as plt
import numpy as np

BLUE= [x/255.0 for x in [0, 114, 178]]
VERMILLION= [x/255.0 for x in [213, 94, 0]]
GREEN= [x/255.0 for x in [0, 158, 115]]
BROWN = [x/255.0 for x in [72, 55, 55]]

cycle = [
'#e41a1c',
'#377eb8',
'#4daf4a',
'#984ea3',
'#ff7f00',
'#ffd92f'
]

d_colors = [
'#777777',
'#2166ac',
'#fee090',
'#fdbb84',
'#fc8d59',
'#e34a33',
'#b30000',
'#00706c'
]

def setPlot(fontsize=24, labelsize=24):
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams["font.size"] = fontsize # 全体のフォントサイズが変更されます
    plt.rcParams['xtick.labelsize'] = labelsize # 軸だけ変更されます
    plt.rcParams['ytick.labelsize'] = labelsize # 軸だけ変更されます

def plotContour(fig, ax, data, title, fontsize, vmin, vmax, cmap):
    ax.set_title(title, fontsize=fontsize)
    t, s = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
    if vmin == None:
        vmin = np.min(data)
    if vmax == None:
        vmax = np.max(data)

    mp = ax.contourf(s, t, np.transpose(data), 15, cmap=cmap, levels=np.linspace(vmin, vmax, 60), extend="both", zorder=-20)
    #fig.colorbar(mp, ax=ax)
    ax.set_rasterization_zorder(-10)
    #ax.set_xlabel(r"Time", fontsize=fontsize)
    return mp
