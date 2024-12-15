from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time
import copy
import matplotlib.pyplot as plt
from src.utils import LossHistory, TrainState, TrainingDisplay, get_loss_fun
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


class cfdDataset(Dataset):
    def __init__(self, x, fx, y):
        self.x = np.array(x, dtype=np.float32)
        self.fx = np.array(fx, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)

    def __len__(self):
        return self.fx.shape[0]

    def __getitem__(self, idx):
        return self.x[idx, :], self.fx[idx, :], self.y[idx, :]


def buildModel(config=None, datadict=None, network=None):
    assert datadict is not None, '{buildModel} datadict not appointed'
    assert network is not None, '{buildModel} network not appointed'

    train_dataset = cfdDataset(x=datadict['treated']['x_train'], fx=datadict['treated']['fx_train'],
                               y=datadict['treated']['y_train'])
    valid_dataset = cfdDataset(x=datadict['treated']['x_valid'], fx=datadict['treated']['fx_valid'],
                               y=datadict['treated']['y_valid'])

    model = Model(train_dataset=train_dataset, valid_dataset=valid_dataset, network=network,
                  model_name=config['model_name'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.compile(lr=config['lr'], metric=config['metric'], loss=config['loss'], device=device)

    return model


def trainModel(config=None, model=None):
    assert isinstance(model, Model), '{trainModel} model not appointed'

    model.train(batch_size=config['batch_size'], epochs=config['epochs'], display_every=100, min_epoch=500)
    model.plot_loss_history()
    return model


def testModel(config=None, model=None, datadict=None):
    assert isinstance(model, Model), '{testModel} model not appointed'
    assert datadict is not None, '{testModel} datadict not appointed'

    model.restore_network()

    x_test_scaled = datadict['treated']['x_test']
    x_test_scaled = torch.tensor(x_test_scaled, dtype=torch.float32, device=model.device)
    fx_test_scaled = datadict['treated']['fx_test']
    fx_test_scaled = torch.tensor(fx_test_scaled, dtype=torch.float32, device=model.device)
    y_test_scaled = datadict['treated']['y_test']

    y_pred_scaled = model.predict(x_test_scaled, fx_test_scaled)
    y_pred_scaled = y_pred_scaled.cpu().numpy()

    scaler = datadict['scaler']['output_scaler']

    y_shape = y_pred_scaled.shape
    num_samples_test = y_shape[0]

    y_pred_scaled = y_pred_scaled.reshape(-1, 1)
    y_test_scaled = y_test_scaled.reshape(-1, 1)

    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test = scaler.inverse_transform(y_test_scaled)

    y_pred = y_pred.reshape(y_shape)
    y_test = y_test.reshape(y_shape)
    y_pred_scaled = y_pred_scaled.reshape(y_shape)
    y_test_scaled = y_test_scaled.reshape(y_shape)

    MSE = np.zeros(num_samples_test)
    MAE = np.zeros(num_samples_test)
    MAPE = np.zeros(num_samples_test)
    # 逐个样本计算评价指标
    for i in range(num_samples_test):
        MSE[i] = mean_squared_error(y_pred[i, :], y_test[i, :])
        MAE[i] = mean_absolute_error(y_pred[i, :], y_test[i, :])
        MAPE[i] = mean_absolute_percentage_error(y_test[i, :], y_pred[i, :])

    metrics = {'MSE': MSE, 'MAE': MAE, 'MAPE': MAPE}  # 预测结果评价指标
    datadict['test_results'] = {'ground_truth_scaled': np.array(y_test_scaled),
                                'predictions_scaled': np.array(y_pred_scaled),
                                'ground_truth': y_test,
                                'predictions': y_pred,
                                'metrics': metrics}

    update_metrics_csv(datadict)

    # 打印日志
    with open('log.txt', 'a+') as f:
        f.write('\n')
        f.write('+' * 50)
        f.write('\n@ MODEL : {}\n'.format(config['model_name']))
        f.write('\n  INPUT FEATURE: {}\n'.format(config['input_feature']))
        f.write('\n  PREDICT FEATURE: {}\n'.format(config['output_feature']))
        for key, value in metrics.items():
            f.write('\n>>> {} evaluation metrics:\n'.format(key))
            f.write('{}: {:.6f}\n'.format(key, np.median(value)))
        f.write('\n')
        f.write('+' * 50)
        f.write('\n')

    for key, value in metrics.items():
        print('\n>>> {} evaluation metrics:\n'.format(key))
        print('{}: {:.6f}\n'.format(key, np.median(value)))
        print('\n')

    return datadict


def update_metrics_csv(config=None, datadict=None):
    # 检查文件是否存在
    metrics_file_path = r'..\results\metrics.csv'
    if os.path.exists(metrics_file_path):
        # 读取CSV文件
        df = pd.read_csv(metrics_file_path)
    else:
        df = pd.DataFrame()

    metrics = datadict['test_results']['metrics']
    for key, value in metrics.items():
        df['{}_{}'.format(config['model_name'], key)] = pd.Series(value)

    # 保存更新后的CSV文件
    df.to_csv(metrics_file_path, index=False)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Model:
    def __init__(self, train_dataset, valid_dataset, network, save_dir=None, model_name='UnnamedModel'):
        assert isinstance(train_dataset,
                          cfdDataset), '{Model init} dataset should be instance of torch.utils.data.Dataset'
        assert isinstance(valid_dataset,
                          cfdDataset), '{Model init} dataset should be instance of torch.utils.data.Dataset'
        assert isinstance(network, nn.Module), '{Model init} network should be instance of torch.nn.Module'

        self.train_dataset = train_dataset  # 训练数据集（结构化）
        self.valid_dataset = valid_dataset  # 验证数据集（结构化）
        self.network = network  # 网络
        self.model_name = model_name  # 模型命名

        if save_dir is None:  # 模型保存目录
            self.save_dir = r'..\results'  # 缺省保存目录
        else:
            self.save_dir = save_dir
        # 检查保存目录是否存在，不存在则创建该目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.opt = None  # 优化器
        self.lr = None  # 初始学习率
        self.batch_size = None  # 批量大小
        self.shuffle = False  # 是否打乱数据
        self.loss_fun = None  # 损失函数
        self.metric_fun = None  # 测试指标
        self.device = None  # 张量计算设备
        self.loss_history = LossHistory()
        self.min_epoch = None  # 最少训练次轮数
        self.best_epoch = 0  # 验证误差最小轮数
        self.min_validate_loss = np.inf  # 最小验证误差
        self.best_model = None  # 最佳训练网络参数

        self.train_state = TrainState()

    def compile(self, lr, loss, metric, device=torch.device('cpu')):
        self.opt = torch.optim.Adam(self.network.parameters(), lr=lr)
        loss_fun = get_loss_fun(loss)
        self.loss_fun = loss_fun
        metric_fun = get_loss_fun(metric)
        self.metric_fun = metric_fun
        self.device = device
        self.network.to(device)

    def train(self, epochs, batch_size=None, shuffle=False, display_every=100, min_epoch=10):
        if batch_size is None:
            # 若未指定批量大小，则整个训练集为一个批量
            self.batch_size = self.train_dataset.__len__()
        else:
            self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_epoch = min_epoch

        # 数据加载器
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        # 训练信息显示器
        training_display = TrainingDisplay()
        self._test(step=0)
        training_display(self.train_state)

        ts = time.time()
        for i in range(1, epochs + 1):
            # 将网络设置为训练模式
            self.network.train()
            # 遍历所有训练数据并更新梯度
            for batch in train_dataloader:
                self.opt.zero_grad()

                x, fx, y = batch
                x = x.to(self.device)
                fx = fx.to(self.device)
                y = y.to(self.device)
                outputs = self.network(x, fx)

                loss_train = self.loss_fun(outputs, y)

                loss_train.backward()
                self.opt.step()

            # 没轮训练结束后在训练集和验证集上计算损失函数和评价指标
            self._test(step=i)

            if i > self.min_epoch and self.loss_history.loss_validate[-1] <= self.min_validate_loss:
                self.best_model = copy.deepcopy(self.network)
                self.min_validate_loss = self.loss_history.loss_validate[-1]
                self.best_epoch = i
                print('best model updated...')

            # 使用字符串格式化输出损失和评价指标
            if i % display_every == 0:
                training_display(self.train_state)
        te = time.time()

        # 日志记录训练时长和最佳迭代轮数
        with open('log.txt', 'a+') as f:
            f.write('\n>>> {} training time: {:.4f}\n'.format(self.model_name, te - ts))
            f.write('\n>>> {} best epoch: {}\n'.format(self.model_name, self.best_epoch))

        # 保存历史最佳模型的训练结果
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        torch.save(self.best_model.state_dict(), f'{self.save_dir}/{self.model_name}.pkl')
        # 将当前模型参数重置为历史最佳模型
        self.network.load_state_dict(self.best_model.state_dict())

    def _test(self, step):
        # 将网络设置为计算模式
        self.network.eval()

        x_train = torch.tensor(self.train_dataset.x, device=self.device)
        fx_train = torch.tensor(self.train_dataset.fx, device=self.device)
        y_train = torch.tensor(self.train_dataset.y, device=self.device)
        x_valid = torch.tensor(self.valid_dataset.x, device=self.device)
        fx_valid = torch.tensor(self.valid_dataset.fx, device=self.device)
        y_valid = torch.tensor(self.valid_dataset.y, device=self.device)

        # 训练集上预测结果
        outputs_train = self.predict(x_train, fx_train)
        # 测试集上预测结果
        outputs_valid = self.predict(x_valid, fx_valid)

        loss_train = self.loss_fun(outputs_train, y_train).item()
        loss_valid = self.loss_fun(outputs_valid, y_valid).item()
        metric_valid = self.metric_fun(outputs_valid, y_valid).item()

        self.loss_history.append(step, loss_train, loss_valid, metric_valid)
        self.train_state.update_train_state(step, loss_train, loss_valid, metric_valid)

    def predict(self, x, fx):
        self.network.eval()

        with torch.no_grad():
            y = self.network(x, fx)

        return y

    def plot_loss_history(self, output_dir=None):
        # 若图形保存目录未指定，则缺省保存到模型保存目录下
        if output_dir is None:
            output_dir = self.save_dir
        plt.figure(figsize=(10, 6))

        plt.semilogy(self.loss_history.steps, self.loss_history.loss_train, label='loss train')
        plt.semilogy(self.loss_history.steps, self.loss_history.loss_validate, label='loss validate')
        plt.semilogy(self.loss_history.steps, self.loss_history.metrics_validate, label='metrics validate')
        plt.xlabel("# Steps")
        plt.title(f"{self.model_name} loss history")
        plt.legend()

        plt.savefig(f'{output_dir}/{self.model_name}_lossHistory.png')

    def restore_network(self, disc_path=None):
        if disc_path is not None:
            # 加载本地的模型参数
            self.network.load_state_dict(torch.load(disc_path))
        elif self.best_model is not None:
            # 加载实例内部的最佳模型参数
            self.network.load_state_dict(self.best_model.state_dict())
        else:
            disc_path = '\\'.join([self.save_dir, self.model_name + '.pkl'])
            self.network.load_state_dict(torch.load(disc_path))

        self.network.to(self.device)
