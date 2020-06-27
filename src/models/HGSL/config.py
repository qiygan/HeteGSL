class HGSL_Config:

    def __init__(self, dataset='cora', gpu=0):
        # ! IDGL configs # Table 7 in paper
        if dataset == 'cora':
            self.lambda_ = 0.9
            self.eta = 0.1  # balance coef. of adj_emb and adj_feat
            self.alpha = 0.2
            self.beta = 0.0
            self.gamma = 0.0  #
            self.epsilon = 0.0  #
            self.num_head = 4  # m: Num of metric heads
            self.delta = 4e-5
        elif dataset == 'citeseer':
            self.lambda_ = 0.6
            self.eta = 0.5  # balance coef. of adj_emb and adj_feat
            self.alpha = 0.4
            self.beta = 0.0
            self.gamma = 0.2  #
            self.epsilon = 0.3  #
            self.num_head = 1  # m: Num of metric heads
            self.delta = 1e-3
        self.T = 10
        # ! Other settings in paper
        self.lr = 0.01  # Fixed lr for all dataset
        self.dropout = 0.5
        self.num_hidden = 16
        self.weight_decay = 5e-4  # Fixed for all dataset
        # ! My config
        # Exp configs
        self.dataset = dataset
        self.gpu = gpu  # -1 to use cpu
        self.out_path = 'results/IDGL/'
        self.activation = 'Relu'
        # Train configs
        self.max_epoch, self.pretrain_epochs = 200, 200
        self.seed = 2020
        self.early_stop = 1
        self.exp_name = f'IDGL_wi.pre_'
        self.ngrl = -1

    def __str__(self):
        # print parameters
        var_list = self.__dict__.keys()
        res = ''
        for var in var_list:
            if var[0] != '_':
                val = self.__dict__[var]
                val_s = "'{}'".format(val) if isinstance(val, str) else str(val)
                res += '--' + var + '=' + val_s + ' '
        return res

