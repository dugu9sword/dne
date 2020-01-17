from luna.program_args import ProgramArgs


class Config(ProgramArgs):
    def __init__(self):
        super().__init__()
        self.task_id = "SST"
        self.finetunable = False

        self.arch = 'bert'

        self.pretrain = 'fasttext'
        self.mode = 'attack'

        self.attack_vectors = 'glove'
        self.attack_tsv = 'nogit/SST-bert-fix.attack.tsv'
        # self.attack_tsv = 'nogit/SST-bert.attack.tsv'
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
        if self.arch in ['bert', 'elmo']:
            model_name = f"{self.task_id}-{self.arch}"
        else:
            model_name =  f"{self.task_id}-{self.arch}-{self.pretrain}"
        if not self.finetunable:
            model_name += '-fix'
        return model_name

    # @property
    # def vocab_name(self):
    #     return f"{self.task_id}-{self.tokenizer}"

    # @property
    # def embedding_weight_name(self):
    #     return f"{self.task_id}-{self.tokenizer}-{self.pretrain}"

    # @property
    # def attack_embedding_weight_name(self):
    #     return f"{self.task_id}-{self.tokenizer}-{self.attack_pretrain}"

    def _check_args(self):
        assert self.arch in ['lstm', 'bert', 'transformer']
        assert self.tokenizer in ['spacy', 'spacyx', 'bert']
        assert self.pretrain in ['glove', 'fasttext', 'random']
        # assert self.mode in ['train', 'evaluate']
        pass