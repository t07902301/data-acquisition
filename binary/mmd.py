import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist
from utils.strategy import *
from utils.set_up import set_up
import utils.statistics.stat_test as stat_test
from sklearn import metrics
def sq_distances(X,Y=None):
    """
    If Y=None, then this computes the distance between X and itself
    """
    assert(X.ndim==2)

    # IMPLEMENT: compute pairwise distance matrix. Don't use explicit loops, but the above scipy functions
    # if X=Y, use more efficient pdist call which exploits symmetry
    if Y is None:
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
    else:
        assert(Y.ndim==2)
        assert(X.shape[1]==Y.shape[1])
        sq_dists = cdist(X, Y, 'sqeuclidean')

    return sq_dists

def gauss_kernel(X, Y=None, sigma=1.0):
    """
    Computes the standard Gaussian kernel k(x,y)=exp(- ||x-y||**2 / (2 * sigma**2))

    X - 2d array, samples on left hand side
    Y - 2d array, samples on right hand side, can be None in which case they are replaced by X
    
    returns: kernel matrix
    """

    # IMPLEMENT: compute squared distances and kernel matrix
    sq_dists = sq_distances(X,Y)
    K = np.exp(-sq_dists / (2 * sigma**2))
    return K

# IMPLEMENT

def quadratic_time_mmd(X,Y,kernel):
    assert X.ndim == Y.ndim == 2
    K_XX = kernel(X,X)
    K_XY = kernel(X,Y)
    K_YY = kernel(Y,Y)
       
    n = len(K_XX)
    m = len(K_YY)
    
    # IMPLEMENT: unbiased MMD statistic (could also use biased, doesn't matter if we use permutation tests)
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    mmd = np.sum(K_XX) / (n*(n-1))  + np.sum(K_YY) / (m*(m-1))  - 2*np.sum(K_XY)/(n*m)
    return mmd

def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def mmd_linear(X, Y):
    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    XY = np.dot(X, Y.T)
    return XX.mean() + YY.mean() - 2 * XY.mean()

def main(epochs,  model_dir ='', device_id=0, base_type=''):
    batch_size, label_map, new_img_num_list, superclass_num, ratio, seq_rounds_config, ds_list, device_config = set_up(epochs, False, device_id)
    clip_processor = Detector.load_clip(device_config)

    # old_data_ratio_list, _ = np.linspace(0, 1, retstep=True, num=5)
    # my_kernel = lambda X,Y : gauss_kernel(X,Y,sigma=0.3)
    my_kernel = lambda X,Y : mmd_rbf(X,Y)

    for epo in range(epochs):
        print('in epoch {}'.format(epo))
        old_model_config = Config.OldModel(batch_size['base'],superclass_num,model_dir, device_config, epo, base_type)
        ds_init = ds_list[epo]
        data_split = Dataset.DataSplits(ds_init, old_model_config.batch_size)
        old_model = Model.prototype_factory(old_model_config.base_type, data_split.loader['train_clip'], clip_processor)
        dataset_gts, dataset_preds, _ = old_model.eval(data_split.loader['test_shift'])
        cor_mask = dataset_gts == dataset_preds
        incor_mask = ~cor_mask
        indices = np.arange(len(data_split.dataset['test_shift']))
        cor_data = torch.utils.data.Subset(data_split.dataset['test_shift'], indices[cor_mask])
        cor_dataloader = torch.utils.data.DataLoader(cor_data, batch_size=old_model_config.batch_size)
        incor_data = torch.utils.data.Subset(data_split.dataset['test_shift'], indices[incor_mask])
        incor_dataloader = torch.utils.data.DataLoader(incor_data, batch_size=old_model_config.batch_size)
        cor_embd, _ = clip_processor.evaluate_clip_images(cor_dataloader)  
        incor_embd, _ = clip_processor.evaluate_clip_images(incor_dataloader)  
        print(mmd_rbf(cor_embd, incor_embd))
        # print(mmd_linear(cor_embd, incor_embd))
        # print(quadratic_time_mmd(cor_embd, incor_embd, my_kernel))
        # print(cor_embd.shape)
        # clf = Detector.SVM(data_split.loader['train_clip'], clip_processor, split_and_search=True)
        # _ = clf.fit(old_model, data_split.loader['val_shift'], 1)
        # cor_dv, incor_dv = stat_test.get_dv_dstr(old_model, data_split.loader['test_shift'], clf)



import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-e','--epochs',type=int,default=1)
    parser.add_argument('-md','--model_dir',type=str,default='')
    parser.add_argument('-d','--device',type=int,default=0)
    parser.add_argument('-bt','--base_type',type=str,default='resnet')

    args = parser.parse_args()
    # method, img_per_cls, Model.save
    main(args.epochs,args.model_dir, args.device, args.base_type)