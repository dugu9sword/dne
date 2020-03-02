import argparse


def string_to_bool(string_val):
    return True if string_val == 'True' else False


class ProgramArgs(argparse.Namespace):
    def __init__(self):
        super(ProgramArgs, self).__init__()
        self.cuda_device = "0"
        
        self.mode = 'evaluate'   # 'evaluate' 'attack' 'augmentation' 'train'
        
        self.workspace = '/disks/sdb/zjiehang/frequency'
        self.train_data_file = '/disks/sdb/zjiehang/frequency/data/agnews/train.csv'
        self.valid_data_file = "/disks/sdb/zjiehang/frequency/data/agnews/valid.csv"
        self.test_data_file = "/disks/sdb/zjiehang/frequency/data/agnews/attack.csv"

        self.embedding_path = '/disks/sdb/zjiehang/DependencyParsing/pretrained_embedding/glove/glove.6B.200d.txt'
        self.cache_path = '/disks/sdb/zjiehang/frequency/cache/vars'

        self.embedding_dim = 200
        self.batch_size = 320
        self.learning_rate = 0.001
        self.hidden_size = 150
        self.epochs = 50
        
        self.crop_batch_size = 30
        self.crop_window_size_rate = 0.3
        self.crop_min_window_size = 3
        
        self.train_data_source = 'crop' # or 'crop'
        self.test_data_source = 'crop' # or 'crop

    @property
    def save_path(self):
        return "{}/saved_models/agnews/{}".format(self.workspace, self.train_data_source)

    @staticmethod
    def parse(verbose=False) -> "ProgramArgs":
        parser = argparse.ArgumentParser()
        default_args = ProgramArgs()
        for key, value in default_args.__dict__.items():
            if type(value) == bool:
                raise Exception("Bool value is not supported!!!")
            parser.add_argument('--{}'.format(key),
                                action='store',
                                default=value,
                                type=type(value),
                                dest=str(key))
        parsed_args = parser.parse_args(namespace=default_args)
        if verbose:
            print("Args:")
            for key, value in parsed_args.__dict__.items():
                print("\t--{}={}".format(key, value))
        assert isinstance(parsed_args, ProgramArgs)
        return parsed_args  # type: ProgramArgs
