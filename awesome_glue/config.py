from luna.program_args import ProgramArgs


class Config(ProgramArgs):
    def __init__(self):
        super().__init__()
        
        # basic settings
        self.task_id = "SST"
        self.finetunable = True
        self.arch = 'lstm'
        self.pretrain = 'glove'      
        self._model_name = "SST-lstm-glove"
#         self._model_name = ""
        self.mode = 'attack'
        
        # transfer settings
        self.adv_data = 'nogit/SST-bert.adv.tsv' 
        
        # training settings
#         self.aug_data = 'nogit/SST-lstm-glove.advaug.tsv'
        self.aug_data = ''

        # attack settings
        self.attack_vectors = 'glove'
        self.attack_data_split = 'dev'
        self.attack_size = 100        
        self.attack_gen_aug = False
        self.attack_gen_adv = False
        
        # other settings
        self.alchemist = False
        self.seed = 2

    @property
    def tokenizer(self):
        if self.arch == 'lstm':
            return 'spacy'
        if self.arch == 'bert':
            return 'bert'

    @property
    def model_name(self):
        if not self._model_name:
            if self.arch in ['bert', 'elmo']:
                model_name = f"{self.task_id}-{self.arch}"
            else:
                model_name =  f"{self.task_id}-{self.arch}-{self.pretrain}"
            if not self.finetunable:
                model_name += '-fix'
            return model_name
        else:
            return self._model_name

    def _check_args(self):
        assert self.arch in ['lstm', 'bert', 'transformer']
        assert self.tokenizer in ['spacy', 'spacyx', 'bert']
        assert self.pretrain in ['glove', 'fasttext', 'random']
        # assert self.mode in ['train', 'evaluate']
        pass
