from luna.program_args import ProgramArgs


class Config(ProgramArgs):
    def __init__(self):
        super().__init__()
        self.task_id = "SST"
        self.finetunable = True

        self.arch = 'lstm'

        self.pretrain = 'glove'
        self.mode = 'attack'

#         self.augment_data = 'nogit/SST-bert.advaug.tsv'
#         self.given_model_name = 'SST-bert-adv'

#         self.augment_data = 'nogit/SST-lstm-glove.advaug.tsv'
        self.augment_data = ''
        self.given_model_name = "SST-lstm-glove-ls1"
#         self.given_model_name = 'SST-bert-adv-ls05'


        self.attack_vectors = 'glove'
        # self.attack_tsv = 'nogit/SST-bert-fix.attack.tsv'
        self.attack_data_split = 'dev'
        self.attack_size = 400
        self.attack_tsv = 'nogit/SST-bert.attack.tsv'
        # self.attack_tsv = 'nogit/SST-lstm-fasttext.attack.tsv'

        # self.layer_noise = 0.0
        # self.embed_noise = 0.0

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
        if not self.given_model_name:
            if self.arch in ['bert', 'elmo']:
                model_name = f"{self.task_id}-{self.arch}"
            else:
                model_name =  f"{self.task_id}-{self.arch}-{self.pretrain}"
            if not self.finetunable:
                model_name += '-fix'
            return model_name
        else:
            return self.given_model_name

    def _check_args(self):
        assert self.arch in ['lstm', 'bert', 'transformer']
        assert self.tokenizer in ['spacy', 'spacyx', 'bert']
        assert self.pretrain in ['glove', 'fasttext', 'random']
        # assert self.mode in ['train', 'evaluate']
        pass
