import numpy as np
import os
import argparse

def get_python_command(dataset, network, keep_prob, max_epochs, optimizer, optim_params, num_exp):
    cmd = 'python tf_run.py '+dataset+' '+network+' '+str(keep_prob)+' '+optimizer+' '+'--max_epochs'+' '+str(max_epochs)#+' '+'--final_run'
    for k, v in optim_params.items():
        cmd += ' '+'--'+k+' '+str(v[num_exp])
    return cmd


def get_optim_params(optimizer):
    optim_params = {}
    if optimizer=='kalpit': # kalpit
        optim_params['momentum'] = 10**np.random.uniform(-3, 0, tot_exp)
        optim_params['max_lrs']  = np.random.uniform(0, 1, tot_exp)
    elif optimizer=='adam': # adam
        optim_params['learning_rate'] = 10**np.random.uniform(-4, 0, tot_exp)
        optim_params[' beta1']         = 1 - 10**np.random.uniform(-3, -1, tot_exp)
        optim_params['beta2']         = 1 - 10**np.random.uniform(-4, -2, tot_exp)
        optim_params['epsilon']       = 10**np.random.uniform(-8, -3, tot_exp)
    elif optimizer=='adadelta': # adadelta
        optim_params['learning_rate'] = 10**np.random.uniform(-4, 0, tot_exp)
        optim_params['rho']           = 1 - np.random.uniform(0, 0.1, tot_exp)
        optim_params['epsilon']       = 10**np.random.uniform(-8, -3, tot_exp)
    return optim_params


def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['mnist', 'cifar10'])
    parser.add_argument('network', choices=['ff', 'conv', 'autoencoder'])
    parser.add_argument('keep_prob', type=float)
    parser.add_argument('optimizer', choices=['sgd', 'adam', 'adadelta', 'kalpit'])
    parser.add_argument('max_epochs', type=int)
    parser.add_argument('tot_exp', type=int)
    return parser.parse_args()

if __name__=='__main__':
    args = ArgumentParser()

    dataset    = args.dataset
    network    = args.network
    keep_prob  = args.keep_prob
    optimizer  = args.optimizer
    max_epochs = args.max_epochs
    tot_exp    = args.tot_exp
    
    optim_params = get_optim_params(optimizer)

    for num_exp in range(tot_exp):
        cmd = get_python_command(dataset, network, keep_prob, max_epochs, optimizer, optim_params, num_exp)
        os.system(cmd)
