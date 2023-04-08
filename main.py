import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import random
import math
from builtins import range
from tqdm.notebook import tqdm

from nn import TwoLayerNet

# 加载数据
def load_data(data_folder):

    files = [
        'train-labels-idx1-ubyte.gz', 
        'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 
        't10k-images-idx3-ubyte.gz'
    ]
    paths = []

    for file in files:
        paths.append(os.path.join(data_folder,file))
    with gzip.open(paths[0], 'rb') as res_path:
        y_train = np.frombuffer(res_path.read(), np.uint8, offset=8).astype("int")
    with gzip.open(paths[1], 'rb') as res_path:
        X_train = np.frombuffer(res_path.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28).astype("float")
    with gzip.open(paths[2], 'rb') as res_path:
        y_test = np.frombuffer(res_path.read(), np.uint8, offset=8).astype("int")
    with gzip.open(paths[3], 'rb') as res_path:
        X_test = np.frombuffer(res_path.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28).astype("float")

    return (X_train, y_train), (X_test, y_test)

def train_val_split(X_train, y_train, X_test, y_test):

    # 划分数据集，抽取随机数
    np.random.seed(1)
    val = list(np.random.choice(60000, 6000, replace=False))
    tra = [i for i in range(60000) if i not in val]

    X_val, y_val = X_train[val], y_train[val]
    X_train, y_train = X_train[tra], y_train[tra]

    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def random_search(input_size, num_classes,
                  X_train, y_train, X_val, y_val):
    # 随机搜索
    random.seed(0)

    results = {}
    best_val_acc = 0
    best_net = None

    best_stats = None
    f = open('log_randomsearch.txt','w')
    for i in range(100):
        hs = random.randint(10,40) * 10
        lr = random.uniform(0.001,0.01)
        reg = random.uniform(0.05,0.15)          
        net = TwoLayerNet(input_size, hs, num_classes)

        # 训练
        stats = net.train(X_train, y_train, X_val, y_val, num_iters = 1500, batch_size = 100,
            learning_rate = lr, learning_rate_decay = 0.95, reg = reg, verbose = False)
        train_acc = (net.predict(X_train) == y_train).mean()
        val_acc = (net.predict(X_val) == y_val).mean()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_net = net 
            best_stats = stats 
            best_hs = hs
            best_net.save_model(model= 'bestmodelparams_randomsearch.npz')     
        results[(hs,lr,reg)] = val_acc
        print ('hidden size: %d learning rate: %e regularization_strength: %e train accuracy: %f validation accuracy: %f' % (hs, lr, reg, train_acc, val_acc))
        f.write('hidden size: %d learning rate: %e regularization_strength: %e train accuracy: %f validation accuracy: %f \n' % (hs, lr, reg, train_acc, val_acc))
    f.close()
    
    print ('Best validation accuracy: %f' % best_val_acc)

    return best_hs, best_stats

def loss_acc_vis(best_stats):
    # 绘制loss曲线
    plt.figure(figsize=(16,6),dpi=300)
    plt.subplot(1, 2, 1)
    plt.plot(best_stats['train_loss_hist'])
    plt.title('Train')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(best_stats['val_loss_hist'])
    plt.title('Val')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.savefig('images/loss.jpg')

    # 绘制acc曲线
    plt.figure(figsize=(16,6),dpi=300)
    plt.subplot(1, 2, 1)
    plt.plot(best_stats['train_acc_hist'])
    plt.title('Train')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(best_stats['val_acc_hist'])
    plt.title('Val')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')

    plt.savefig('images/acc.jpg')

def visualize_grid(Xs, ubound=255.0, padding=1):

    (N, H, W, C) = Xs.shape
    grid_size = int(math.ceil(math.sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding

    return grid

def show_net_weights(W1):

    W1 = W1.reshape(28, 28, 1, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.savefig('images/w1_vis.jpg')

def w1_w2_vis():
    # 可视化参数W1
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    params = {}
    p = np.load('bestmodelparams_randomsearch.npz')
    params['W1'] = p['w1']
    params['W2'] = p['w2']
    params['b1'] = p['b1']
    params['b2'] = p['b2']
    show_net_weights(params['W1'])

    ## 可视化W2
    plt.imshow(params['W2'].transpose(1,0))
    plt.gca().axis('off')
    plt.savefig('images/w2_vis.jpg')

if __name__=='__main__':

    (X_train, y_train), (X_test, y_test)= load_data('MINIST_data/')

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_split(X_train, y_train, X_test, y_test)

    input_size = 28 * 28
    hidden_size = 100
    num_classes = 10

    best_hs, best_stats = random_search(input_size, num_classes, X_train, y_train, X_val, y_val)

    # 可视化loss和acc曲线
    loss_acc_vis(best_stats)

    # 可视化网络参数
    w1_w2_vis()
    
    # 对test进行预测
    net = TwoLayerNet(input_size, best_hs, num_classes)
    net.load_model('bestmodelparams_randomsearch.npz')
    test_acc = (net.predict(X_test) == y_test).mean()
    print ('test accuracy: %f' % test_acc)

    