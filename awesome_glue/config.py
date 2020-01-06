from luna.program_args import ProgramArgs


class Config(ProgramArgs):
    def __init__(self):
        super().__init__()
        self.task_id = "CoLA"
        self.pretrain = "bert"
        self.finetunable = True
        self.mode = 'train'

        self.bert_noise = 0.0
        self.embed_noise = 0.0
        self.lstm_noise = 0.0

        self.alchemist = False


    def _check_args(self):
        # assert self.mode in ['train', 'evaluate']
        pass