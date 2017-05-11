import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_loss(losses, save_dir, fname, first_use=False):
    if first_use:
        f = open(os.path.join(save_dir, fname), 'w')
    else:
        f = open(os.path.join(save_dir, fname), 'a')
    for loss in losses:
        f.write(str(loss)+'\n')
    f.close()
    return


def plot_loss(losses, save_dir, plotname, title=''):
    plt.figure()
    plt.semilogy(losses)
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.title(title)
    plt.savefig(os.path.join(save_dir, 'plot_'+plotname))
    return
