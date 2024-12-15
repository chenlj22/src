from src.dataset import loadRawData, pretreatment
from src.network import buildNetwork
from src.model import buildModel, trainModel, testModel
from src.post import visualization


"""
本代码框架参考 deepxde
"""
# ----------------------------------------------------------------------------------------------------------------------

config = {
    
    # 数据相关参数
    # "data_dir": r"..\data",  # 数据根目录，需修改为自己的data文件夹地址
    "data_dir": r"C:\Users\26382\Desktop\2024fall\explore\Group-3-Condensor\data",
    "train_dataset": '20241123-2d2p-poroZoneOpt-train',
    "valid_dataset": '20241123-2d2p-poroZoneOpt-valid',
    "test_dataset": '20241123-2d2p-poroZoneOpt-test',
    "input_feature": "porosity_field",  # 输入特征
    "output_feature": "Ug_vec",  # 输出特征
    
    "fun_dim": 1,  # 输入特征维度
    "out_dim": 2,  # 输出特征维度

    "num_element": 1902, # 网格数

    # Transolver超参数
    "activation": "gelu",  # 激活函数类型
    "optimizer": "adam",  # 优化器类型
    "lr": 0.001,  # 学习率
    "epochs": 2000,  # 训练次数
    "batch_size": 32,  # 批量规模
    "loss": "MSE",  # 训练损失函数类型
    "metric": 'MSE',  # 验证损失函数类型
    
    # MLP 结构(固定隐藏层大小)
    "n_input": 9,
    "n_hidden": 512, # 隐藏层大小
    "n_output": 1902*2,
    "n_layers": 2, # 隐藏层层数
    

    # 模型名称
    "model": 'MLP',
    "model_name": "poro2Ug_MLP"  # 模型名称
}


"""场景控制"""
TRAIN = True  # 训练
TEST = True  # 预测
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # 全局字典 datadict 用于存储数据
    datadict = {}

    # 读取原始数据
    datadict = loadRawData(config=config, datadict=datadict)

    # 数据预处理
    datadict = pretreatment(config=config, datadict=datadict)

    # 搭建神经网络
    network = buildNetwork(config=config)

    # 创建模型
    model = buildModel(config=config, datadict=datadict, network=network)

    if TRAIN:
        # 训练模型
        model = trainModel(config=config, model=model)

    if TEST:
        # 测试模型
        datadict = testModel(config=config, model=model, datadict=datadict)

        visualization(config=config, datadict=datadict)
