from luna.program_args import ProgramArgs


class Config(ProgramArgs):
    def __init__(self):
        super().__init__()
        self.task_id = "RTE"
        self.finetunable = True

        self.tokenizer = 'bert'
        self.arch = 'bert'

        self.pretrain = 'fasttext_ol'
        self.mode = 'train'

        self.attack_tsv = 'nogit/SST-lstm-random.attack.tsv'

        # self.layer_noise = 0.0
        # self.embed_noise = 0.0

        self.alchemist = False
        self.seed = 2

    @property
    def model_name(self):
        return f"{self.task_id}-{self.arch}-{self.pretrain}"


    def _check_args(self):
        assert self.arch in ['lstm', 'bert', 'transformer']
        assert self.tokenizer in ['spacy', 'spacyx', 'bert']
        assert self.pretrain in ['glove', 'fasttext', 'sgns', 'fasttext_ol', 'random']
        # assert self.mode in ['train', 'evaluate']
        pass