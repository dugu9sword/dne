from luna.program_args import ProgramArgs


class Config(ProgramArgs):
    def __init__(self):
        super().__init__()
        self.pretrain = "bert"
        self.fix_embed = False
        self.mode = 'attack'

        self.bert_noise = 0.0
        self.embed_noise = 0.0
        self.lstm_noise = 0.0

        self.alchemist = False


    def _check_args(self):
        # assert self.mode in ['train', 'evaluate']
        pass