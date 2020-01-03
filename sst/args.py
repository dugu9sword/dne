import argparse


def string_to_bool(string_val):
    return True if string_val == 'True' else False


class ProgramArgs(argparse.Namespace):
    def __init__(self):
        super(ProgramArgs, self).__init__()
        self.cuda_device = "2"
        
        self.mode = 'attack'   # 'evaluate' 'attack' 'augmentation'
        
        self.workspace = '/disks/sdb/zjiehang/frequency'
        self.data = 'SST'
        self.train_data_file = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt"
        self.dev_data_file = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt"
        self.test_data_file = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/test.txt"

        self.epoch = 128
        self.hidden_dim = 256
        self.batch_size = 32
        self.learning_rate = 1e-1
        self.min_count = 0
        self.num_layers = 2
        self.optimizer = 'sgd'  # 'sparse' 'sgd' 'adam' 
        
        self.embedding_trainable = 'True'
        self.pretrain = 'fasttext'
        
        self.dropout_rate = 0.0
        self.weight_decay = 0.0
        
        # for noise
        self.embed_noise = 0.0
        self.lstm_noise = 0.0# when noise is 0.0, meaning not add noise
        
    @property
    def cache_path(self):
        return "{}/cache/vars".format(self.workspace)

    @property
    def cache_prefix(self):
        return "sst_elmo" if self.pretrain == 'elmo' else 'sst'
    
    @property
    def saved_path(self):
        return '{}/saved_models/{}/{}_{}_{}_{}noise'.format(self.workspace,
                                                            self.data, 
                                                            self.pretrain, 
                                                            'train' if self.is_embedding_trainable else 'not_train',
                                                            self.optimizer,
                                                            self.embed_noise)
    
    @property
    def is_embedding_trainable(self):
        return string_to_bool(self.embedding_trainable)

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
