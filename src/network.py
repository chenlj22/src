from src.Transolver import MLP

'''def buildNetwork(config=None):
    """搭建神经网络"""
    network_dict = {
        'Transolver': Transolver
    }

    network = network_dict[config['model']](
        space_dim=2,
        n_layers=config['n_layers'],
        n_hidden=config['n_hidden'],
        dropout=config['dropout'],
        n_head=config['n_head'],
        act='gelu',
        mlp_ratio=1,
        fun_dim=config['fun_dim'],
        out_dim=config['out_dim'],
        slice_num=config['slice_num'],
        ref=8,
        unified_pos=False
    )

    return network'''

def buildNetwork(config=None):
    """搭建MLP神经网络"""
    network_dict = {
        'MLP': MLP
    }

    network = network_dict[config['model']](
        n_input = config['n_input'],
        n_hidden = config['n_hidden'],
        n_output = config['n_output'],
        n_layers = config['n_layers'],
        act = config['activation'],
        res = True,
    )

    return network
