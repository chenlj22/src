import os.path
import pickle
import numpy as np
from sklearn.preprocessing import QuantileTransformer


def loadDataset(data_dir, dataset_type: str = None, iFeature=None, oFeature=None):
    """从磁盘读取数据集并选取特征"""
    data_save_path = '\\'.join([data_dir, dataset_type + '.pkl'])

    """ CFD 数据集 dataset 为字典 """
    with open(data_save_path, 'rb') as f:
        datadict = pickle.load(f)

    x = []
    y = []

    for i, key in enumerate(datadict.keys()):
        # 选取输入数据
        if iFeature is not None:
            x.append(datadict[key][iFeature])
        # 选取输出数据
        if oFeature is not None:
            y.append(datadict[key][oFeature])

    return np.array(x), np.array(y)


def loadRawData(config=None, datadict=None):
    """读取训练集，验证集，测试集"""
    assert datadict is not None, '{loadDataset} datadict not appointed'

    fx_train, y_train = loadDataset(config['data_dir'], config['train_dataset'], iFeature=config['input_feature'],
                                    oFeature=config['output_feature'])
    fx_valid, y_valid = loadDataset(config['data_dir'], config['valid_dataset'], iFeature=config['input_feature'],
                                    oFeature=config['output_feature'])
    fx_test, y_test = loadDataset(config['data_dir'], config['test_dataset'], iFeature=config['input_feature'],
                                  oFeature=config['output_feature'])

    # 特征维度为1时手动增加特征维度
    if config['fun_dim'] == 1:
        fx_train = fx_train[:, :, np.newaxis]
        fx_valid = fx_valid[:, :, np.newaxis]
        fx_test = fx_test[:, :, np.newaxis]

    if config['out_dim'] == 1:
        y_train = y_train[:, :, np.newaxis]
        y_valid = y_valid[:, :, np.newaxis]
        y_test = y_test[:, :, np.newaxis]

    # 空间坐标
    x_train, _ = loadDataset(config['data_dir'], config['train_dataset'], iFeature='element_centroids')
    x_valid, _ = loadDataset(config['data_dir'], config['valid_dataset'], iFeature='element_centroids')
    x_test, _ = loadDataset(config['data_dir'], config['test_dataset'], iFeature='element_centroids')

    raw = {'x_train': x_train, 'fx_train': fx_train, 'y_train': y_train,
           'x_valid': x_valid, 'fx_valid': fx_valid, 'y_valid': y_valid,
           'x_test': x_test, 'fx_test': fx_test, 'y_test': y_test}

    # 保存原始输入输出数据
    datadict['raw'] = raw

    return datadict


def pretreatment(config=None, datadict=None):
    """特征缩放处理"""
    assert datadict is not None, '{pretreatment} datadict not appointed'

    # 空间坐标
    x_train = datadict['raw']['x_train']
    x_valid = datadict['raw']['x_valid']
    x_test = datadict['raw']['x_test']

    # 输入特征孔隙率本身在 0,1 之间，暂不做处理
    fx_train = datadict['raw']['fx_train']
    fx_valid = datadict['raw']['fx_valid']
    fx_test = datadict['raw']['fx_test']

    # 输出特征缩放处理
    y_train_raw = datadict['raw']['y_train']
    y_valid_raw = datadict['raw']['y_valid']
    y_test_raw = datadict['raw']['y_test']

    y_train_shape = y_train_raw.shape
    y_valid_shape = y_valid_raw.shape
    y_test_shape = y_test_raw.shape

    # 初始化缩放器，从训练集中计算缩放参数
    output_scaler = QuantileTransformer(output_distribution='uniform', random_state=42)

    y_train = output_scaler.fit_transform(y_train_raw.reshape(-1, 1))
    y_valid = output_scaler.transform(y_valid_raw.reshape(-1, 1))
    y_test = output_scaler.transform(y_test_raw.reshape(-1, 1))

    if not os.path.exists(r'..\results'):
        os.mkdir(r'..\results')
    # 保存 scaler 到本地文件
    with open(r'..\results\{}_OScaler.pkl'.format(config['model_name']), 'wb') as file:
        pickle.dump(output_scaler, file)
    with open(r'..\results\{}_OScaler.pkl'.format(config['model_name']), 'rb') as file:
        output_scaler_restore = pickle.load(file)

    y_train = y_train.reshape(y_train_shape)
    y_valid = y_valid.reshape(y_valid_shape)
    y_test = y_test.reshape(y_test_shape)

    # 输入
    treated = {'x_train': x_train, 'fx_train': fx_train, 'y_train': y_train,
               'x_valid': x_valid, 'fx_valid': fx_valid, 'y_valid': y_valid,
               'x_test': x_test, 'fx_test': fx_test, 'y_test': y_test}

    # 保存预处理后的输入输出数据
    datadict['treated'] = treated
    # 保存缩放器
    datadict['scaler'] = {'output_scaler': output_scaler, 'input_scaler': None}

    return datadict
