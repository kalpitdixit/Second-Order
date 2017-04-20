import theano
import nump as np


def get_images_and_labels(data_dir):
    ## images
    with open(os.path.join(data_dir, 'images', 'info'), 'r') as f:
        num_images = int(f.readline().strip())
    images = np.zeros((num_imags, 28, 28))
    for i in range(num_images):
        images[i,:,:] = np.load(os.path.join(data_dir, 'images', 'image_{}.npy'.format(i)))
    ## labels
    labels = []
    with open(os.path.join(data_dir, 'labels', 'labels.txt'), 'r') as f:
        for line in f:
            labels.append(int(line.strip()))
    return images, labels

def get_data(data_dir):
    """
    returns a dict called "data"
    data['train_images'] - 3d np array. data['train_images'][i] refers to train image number i. 2nd x 3rd dm is 28x28.
    data['train_labels'] - 1d list. data['train_labels'][i] refers to label of train image number i.
    data['test_images'] - 3d np array. data['test_images'][i] refers to test image number i. 2nd x 3rd dm is 28x28.
    data['test_labels'] - 1d list. data['test_labels'][i] refers to label of test image number i.
    """

    data = {}
    ### training 
    train_images, train_labels = get_images_and_labels(os.path.join(data_dir, 'training'))
    data['train_images'] = train_images
    data['train_labels'] = train_labels
    ### testing 
    test_images, test_labels = get_images_and_labels(os.path.join(data_dir, 'testing'))
    data['test_images'] = test_images
    data['test_labels'] = test_labels
    return data
 
if __name__=="__main__":
    data_dir = '/atlas/u/kalpit/data'
    data = get_data(data_dir)
