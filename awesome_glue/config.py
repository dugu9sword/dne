from luna.program_args import ProgramArgs


class Config(ProgramArgs):
    def __init__(self):
        super().__init__()

        # basic settings
        self.task_id = "SST"
        self.finetunable = True
        self.arch = 'lstm'
        self.pretrain = 'glove'
        # self._model_name = "SST-lstm-adv"
        self._model_name = ""
        self.mode = 'train'

        # transfer settings
        self.adv_data = 'nogit/AGNEWS-lstm.hotflip.adv.tsv'

        # training settings
        #         self.aug_data = 'nogit/SST-lstm-glove.advaug.tsv'
        self.aug_data = ''
        self.adv_iter = 2
        self.adv_replace_num = 5
        self.adv_constraint = True

        # predictor settings
        self.pred_ensemble = 1
        self.pred_transform = ''
        self.pred_transform_args = 0.3

        # attack settings
        self.attack_method = 'pwws'
        self.attack_vectors = 'counter'
        self.attack_data_split = 'dev'
        self.attack_size = 200
        self.attack_gen_aug = False
        self.attack_gen_adv = False

        # other settings
        self.alchemist = False
        self.seed = 2

    @property
    def tokenizer(self):
        if self.arch in ['lstm', 'glstm']:
            return 'spacy'
        if self.arch == 'bert':
            return 'bert'

    @property
    def model_name(self):
        if not self._model_name:
            #             if self.arch in ['bert', 'elmo']:
            #                 model_name = f"{self.task_id}-{self.arch}"
            #             else:
            #                 model_name =  f"{self.task_id}-{self.arch}-{self.pretrain}"
            #             if not self.finetunable:
            #                 model_name += '-fix'
            #             return model_name
            model_name = f"{self.task_id}-{self.arch}"
            if self.adv_iter != 0:
                model_name += f'-adv.{self.adv_iter}.{self.adv_replace_num}'
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
