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


def get_save_dir(dataset, network, cfg):
    save_dir = os.path.join(os.getcwd(), 'results', dataset, network, cfg.optimizer)
    nfiles_fname = os.path.join(save_dir,'num_files')
    if os.path.exists(nfiles_fname):
        with open(nfiles_fname, 'r') as f:
            nfiles = int(f.readline().strip())
    else:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        nfiles = 0
    nfiles += 1
    while os.path.exists(os.path.join(save_dir, 'run_'+str(nfiles))):
        nfiles += 1
    with open(nfiles_fname, 'w') as f:
        f.write(str(nfiles))
    save_dir = os.path.join(save_dir, 'run_'+str(nfiles))
    os.makedirs(save_dir)
    print 'save_dir: ', save_dir
    return save_dir
