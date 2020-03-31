from luna.program_args import ProgramArgs


class Config(ProgramArgs):
    def __init__(self):
        super().__init__()

        # basic settings
        self.task_id = "SNLI"
        self.embed = ''   # d/g/_
        self.arch = 'esim'
        self.pretrain = 'glove'
#         self._model_name = "AGNEWS-lstm-hot.1.5.con"
        self._model_name = ""   # if set to tmp, existing models will be overrided
        self.mode = 'train'
        
        # dirichlet settings
        self.dir_temp = 5.0
        
        # graph settings
        self.gnn_type = 'mean'
        self.gnn_hop = 1

        # training settings
        # self.aug_data = 'nogit/AGNEWS-lstm.pwws.aug.tsv'
        self.aug_data = ''
        self.adv_iter = 1
        self.adv_policy = 'hot'    # hot -> hotflip, rdm -> random
        self.adv_replace_num = 5
        self.adv_constraint = True

        # predictor settings
        self.pred_ensemble = 1
        self.pred_transform = ""
        self.pred_transform_args = ""

        # attack settings
        self.attack_method = 'pwws'
        self.attack_vectors = 'counter'
        self.attack_data_split = 'dev'
        self.attack_size = 200
        # self.attack_data_split = 'train'
        # self.attack_size = -1
        self.attack_gen_adv = False

        # transfer settings
        self.adv_data = 'nogit/AGNEWS-lstm.hotflip.adv.tsv'

        # other settings
        self.alchemist = False
        self.seed = 2
        self.cuda = 3

    @property
    def tokenizer(self):
        if self.arch == 'bert':
            return 'bert'
        else:
            return 'spacy'

    @property
    def model_name(self):
        if not self._model_name:
            model_name = f"{self.task_id}-{self.embed + self.arch}"
            assert not (self.aug_data != '' and self.adv_iter != 0)
            if self.aug_data != '':
                model_name += '-aug'
            if self.embed == 'd':
                model_name += f'-{self.dir_temp}'
            if self.embed == 'g':
                model_name += f'-{self.gnn_type}-{self.gnn_hop}'
            if self.adv_iter != 0:
                model_name += f'-{self.adv_policy}.{self.adv_iter}.{self.adv_replace_num}'
                if self.adv_constraint:
                    model_name += '.con'
                else:
                    model_name += '.unc'
            return model_name
        else:
            return self._model_name

    def _check_args(self):
        pass
        # assert self.arch in ['glstm', 'lstm', 'bert', 'mlstm', "dlstm", "cnn"]
        # assert self.tokenizer in ['spacy', 'bert']
        # assert self.pretrain in ['glove', 'fasttext']
        # assert self.mode in ['train', 'evaluate']
