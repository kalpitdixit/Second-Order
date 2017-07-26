import os
import argparse


def read_config(fname):
    config = {}
    with open(os.path.join(fname, 'config.txt'), 'r') as f:
        for line in f:
            line = [x.strip() for x in line.strip().split(':')]
            config[line[0]] = line[1]
    return config


def read_val_files(fname):
    best_val_loss = float('inf')
    with open(os.path.join(fname, 'validation_cost.txt'), 'r') as f:
        for line in f:
            best_val_loss = min(best_val_loss, float(line.strip()))
    return str(best_val_loss)


def read_result(fname, run_num):
    result = {'run_num': str(run_num)}
    config = read_config(fname)
    result.update(config)

    result['val_loss'] = read_val_files(fname)
    return result


def get_print_keys(result, optimizer):
    keys = []
    if optimizer=='sgd':
        keys.extend(['learning_rate', 'momentum', 'use_nesterov'])
    elif optimizer=='adam':
        keys.extend(['learning_rate', 'beta1', 'beta2', 'epsilon'])
    elif optimizer=='adadelta':
        keys.extend(['learning_rate', 'rho', 'epsilon'])
    elif optimizer=='kalpit':
        keys.extend(['momentum', 'max_lr'])
    keys.append('val_loss')
    keys.append('run_num')
    return keys


def print_info(result, optimizer):
    print_keys = get_print_keys(result, optimizer)
    print '  '.join([k+': '+'{:20s}'.format(result[k]) for k in print_keys])
    return

    
def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['mnist', 'cifar10'])
    parser.add_argument('network', choices=['ff', 'conv', 'autoencoder'])
    parser.add_argument('optimizer', choices=['sgd', 'adam', 'adadelta', 'kalpit'])
    parser.add_argument('start_num', type=int)
    parser.add_argument('end_num', type=int)
    return parser.parse_args()


if __name__=='__main__':

    args = ArgumentParser()

    dirname = os.path.join('results', args.dataset, args.network, args.optimizer)
    start_run_num = args.start_num
    end_run_num = args.end_num

    ## read results
    best_val_loss = float('inf')
    best_run_num = -1
    for i in range(start_run_num, end_run_num+1):
        result = read_result(os.path.join(dirname,'run_'+str(i)), i)
        print_info(result, args.optimizer)
        
        if best_val_loss > float(result['val_loss']):
            best_val_loss = float(result['val_loss'])
            best_run_num = i

    ## print
    print 'Best Val Loss: ', best_val_loss, '  run number: ', best_run_num
