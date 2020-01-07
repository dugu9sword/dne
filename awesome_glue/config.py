from luna.program_args import ProgramArgs


class Config(ProgramArgs):
    def __init__(self):
        super().__init__()
        self.task_id = "RTE"
        self.finetunable = True
        self.mode = 'train'

        self.layer_noise = 0.3
        self.embed_noise = 0.3

        self.alchemist = False


    def _check_args(self):
        # assert self.mode in ['train', 'evaluate']
        pass