import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib  import cm
from mpl_toolkits import mplot3d

from matplotlib import colors

class QResults():
    def __init__(self):
        model_name = ''
        rmnse_avg_test = 0
        rmnse_avg_train = 0
        n_pred_005_avg_test = 0
        n_pred_005_avg_train = 0
        n_pred_050_avg_test = 0
        n_pred_050_avg_train = 0
        avg_train_vpt = 0
        avg_test_vpt = 0

    def info(self):
        print('Model: {}'.format(self.model_name))
        print('TEST: rmnse={}, pred_005={}, pred_05={}, vpt={}'.format(self.rmnse_avg_test, self.n_pred_005_avg_test, self.n_pred_050_avg_test, self.avg_test_vpt))
        print('TRAIN: rmnse={}, pred_005={}, pred_05={}, vpt={}'.format(self.rmnse_avg_train, self.n_pred_005_avg_train, self.n_pred_050_avg_train, self.avg_train_vpt))

def out_qslist_tofile(qslist, filepath):
    strls = ['#Rank #\t #Name #\t #Test_050 #\t #Test_005 #\t #Test_rmnse #\t #Test_vpt #\t #Train_050 #\t #Train_005 #\t #Train_rmnse #\t #Train_vpt']
    for i in range(len(qslist)):
        qs = qslist[i]
        strls.append('#{} #\t #{} #\t #{:.1f} #\t #{:.1f} #\t #{:.3f} #\t #{:.3f} #\t #{:.1f} #\t #{:.1f} #\t #{:.3f} #\t #{:.3f}'.format(\
            i, qs.model_name, qs.n_pred_050_avg_test, qs.n_pred_005_avg_test, qs.rmnse_avg_test, qs.avg_test_vpt,\
            qs.n_pred_050_avg_train, qs.n_pred_005_avg_train, qs.rmnse_avg_train, qs.avg_train_vpt
        ))
    with open(filepath, mode='w') as fw:
        fw.write('\n'.join(strls))


def calVPT(pred, truth, eps = 0.5, dt = 0.01, maxLyp = 0.9056):
    assert(pred.shape == truth.shape)
    N, M = pred.shape
    vpt = 0
    sigma = np.std(truth[:, :], axis=0)
    sigma2 = np.square(sigma)
    for i in range(N):
        diff = pred[i, :] - truth[i, :]
        diff2 = np.square(diff)
        mse  = np.mean(diff2 / sigma2)
        rmse = np.sqrt(mse)
        if (rmse > eps):
            break
        else:
            vpt = i
        vpt = (i * dt) / maxLyp
    return vpt

def calNRMSE(pred, truth):
    assert(pred.shape == truth.shape)
    N, M = pred.shape
    sigma = np.std(truth[:, :], axis=0)
    sigma2 = np.square(sigma)
    rs = []
    for i in range(N):
        diff = pred[i, :] - truth[i, :]
        diff2 = np.square(diff)
        mse  = np.mean(diff2 / sigma2)
        rmse = np.sqrt(mse)
        rs.append(rmse)
    return rs

def createTestingContours(save_path, target, output, dt, maxLyp, ic_idx, set_name):
    fontsize = 12
    error = np.abs(target-output)
    # vmin = np.array([target.min(), output.min()]).min()
    # vmax = np.array([target.max(), output.max()]).max()
    vmin = target.min()
    vmax = target.max()
    vmin_error = 0.0
    vmax_error = target.max()

    print("VMIN: {:} \nVMAX: {:} \n".format(vmin, vmax))

    # Plotting the contour plot
    fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(12, 6), sharey=True)
    fig.subplots_adjust(hspace=0.4, wspace = 0.4)

    axes[0].set_ylabel(r"Time $t$", fontsize=fontsize)
    createContour_(fig, axes[0], target, "Target", fontsize, vmin, vmax, plt.get_cmap("seismic"), dt, maxLyp)
    createContour_(fig, axes[1], output, "Output", fontsize, vmin, vmax, plt.get_cmap("seismic"), dt, maxLyp)
    createContour_(fig, axes[2], error, "Error", fontsize, vmin_error, vmax_error, plt.get_cmap("Reds"), dt, maxLyp)
    for ftype in ['png']:
        fig_path = save_path + "/prediction_{:}_{:}_contour.{}".format(set_name, ic_idx, ftype)
        plt.savefig(fig_path)
    plt.close()

def createContour_(fig, ax, data, title, fontsize, vmin, vmax, cmap, dt, maxLyp):
    ax.set_title(title, fontsize=fontsize)
    t, s = np.meshgrid(np.arange(data.shape[0])* dt / maxLyp, np.arange(data.shape[1]))
    mp = ax.contourf(s, t, np.transpose(data), 15, cmap=cmap, levels=np.linspace(vmin, vmax, 60), extend="both", zorder=-20)
    fig.colorbar(mp, ax=ax)
    ax.set_rasterization_zorder(-10)
    ax.set_xlabel(r"$State$", fontsize=fontsize)
    return mp