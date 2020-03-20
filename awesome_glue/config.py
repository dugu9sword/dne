from luna.program_args import ProgramArgs


class Config(ProgramArgs):
    def __init__(self):
        super().__init__()

        # basic settings
        self.task_id = "AGNEWS"
        self.finetunable = True
        self.arch = 'lstm'
        self.pretrain = 'glove'
        # self._model_name = "SST-lstm-adv"
        self._model_name = ""
        self.mode = 'attack'

        # training settings
        # self.aug_data = 'nogit/AGNEWS-lstm.pwws.aug.tsv'
        self.aug_data = ''
        self.adv_iter = 0
        # hot -> hotflip, rdm -> random
        self.adv_policy = 'rdm'
        self.adv_replace_num = 20
        self.adv_constraint = True

        # predictor settings
        self.pred_ensemble = 1
        self.pred_transform = ''
        self.pred_transform_args = ''

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
        self.cuda = 0

    @property
    def tokenizer(self):
        if self.arch in ['lstm', 'glstm']:
            return 'spacy'
        if self.arch == 'bert':
            return 'bert'

    @property
    def model_name(self):
        if not self._model_name:
            model_name = f"{self.task_id}-{self.arch}"
            assert not (self.aug_data != '' and self.adv_iter != 0)
            if self.aug_data != '':
                model_name += '-aug'
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
        assert self.arch in ['glstm', 'lstm', 'bert']
        assert self.tokenizer in ['spacy', 'bert']
        assert self.pretrain in ['glove', 'fasttext', 'random']
        # assert self.mode in ['train', 'evaluate']
