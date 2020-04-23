from luna.program_args import ProgramArgs


class Config(ProgramArgs):
    def __init__(self):
        super().__init__()

        # basic settings
        self.task_id = "IMDB"
        self.embed = ''   # d/g/w/_
        self.arch = 'boe'
        self._pool = 'max'
        self.pretrain = 'glove'
#         self._model_name = "AGNEWS-lstm-hot.1.5.con"
        self._model_name = ""   # if set to tmp, existing models will be overrided
        self.mode = 'attack'
        
        # dirichlet settings
        self.dir_alpha = 0.1
        self.nbr_num = 24
        self.nbr_2nd = '11'
    
        # graph settings
        self.gnn_type = 'mean'
        self.gnn_hop = 1

        # training settings
        # self.aug_data = 'nogit/AGNEWS-lstm.pwws.aug.tsv'
        self.aug_data = ''
        self.adv_iter = 1
        # hot -> hotflip, rdm -> random, diy -> model do it itself
        self.adv_policy = 'hot'
        self.adv_step = 10.0
        self.adv_replace_num = 0.15

        # predictor settings
        self.pred_ensemble = 16
        self.pred_transform = ""
        self.pred_transform_args = ""

        # attack settings
        self.attack_method = 'genetic'
        self.attack_use_lm = False
        self.attack_data_split = 'test'
        self.attack_size = 200
        # self.attack_data_split = 'train'
        # self.attack_size = -1
        self.attack_gen_adv = False

        # transfer settings
        self.adv_data = ''

        # other settings
        self.alchemist = False
        self.seed = 2
        self.cuda = 0

    @property
    def pool(self):
        if not self._pool:
            if self.arch == 'cnn':
                return 'max'
            elif self.arch == 'boe':
                return 'mean'
        else:
            return self._pool

    @property
    def tokenizer(self):
        if self.arch == 'bert':
            return 'bert'
        else:
            return 'spacy'

    @property
    def model_name(self):
        if not self._model_name:
            model_name = f"{self.task_id}-{self.embed + self.arch}-{self.pretrain}-{self.pool}"
            assert not (self.aug_data != '' and self.adv_iter != 0)
            if self.aug_data != '':
                model_name += '-aug'
            if self.embed in ['d', 'w']:
                model_name += f'-{self.dir_alpha}'
                if self.nbr_2nd[0] == '2':
                    model_name += '-2nd'
            if self.embed == 'g':
                model_name += f'-{self.gnn_type}-{self.gnn_hop}'
            if self.adv_iter != 0:
                if self.adv_policy in ['hot', 'rdm']:
                    model_name += f'-{self.adv_policy}.{self.adv_iter}.{self.adv_replace_num}'
                elif self.adv_policy == 'diy':
                    model_name += f'-{self.adv_policy}.{self.adv_iter}.{self.adv_step}'
            return model_name
        else:
            return self._model_name

    def _check_args(self):
        pass
        # assert self.arch in ['glstm', 'lstm', 'bert', 'mlstm', "dlstm", "cnn"]
        # assert self.tokenizer in ['spacy', 'bert']
        # assert self.pretrain in ['glove', 'fasttext']
        # assert self.mode in ['train', 'evaluate']
