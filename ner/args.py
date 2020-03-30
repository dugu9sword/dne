import argparse

# 
# def __init__(self):
#     super(ProgramArgs, self).__init__()
#     self.cuda_device = "0"
# 
#     self.mode = 'train'  # 'evaluate' 'attack' 'augmentation'
# 
#     self.workspace = '/disks/sdb/zjiehang/frequency'
#     self.data = 'conll2003'
#     self.label_encoding = 'BIOUL'
#     self.train_data_file = "/disks/sdb/zjiehang/frequency/data/conll2003/train.txt"
#     self.dev_data_file = "/disks/sdb/zjiehang/frequency/data/conll2003/dev.txt"
#     self.test_data_file = "/disks/sdb/zjiehang/frequency/data/conll2003/test.txt"
# 
#     self.embedding_type = 'elmo'  # 'elmo' or 'bert'
# 
#     # for token(word) embedding
#     self.token_embedding_dim = 100
#     self.embedding_file = "/disks/sdb/zjiehang/DependencyParsing/pretrained_embedding/glove/glove.6B.100d.txt"
# 
#     # for char embedding (based on cnn-char encoder)
#     # see: End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF
#     self.char_embedding_dim = 16
#     self.num_filter = 128
#     self.ngram_filter_sizes = [3]
# 
#     # for emlo embedding
#     self.elmo_options_file = "/disks/sdb/zjiehang/frequency/pretrained_embedding/elmo/options.json"
#     self.elmo_weight_file = "/disks/sdb/zjiehang/frequency/pretrained_embedding/elmo/weights.hdf5"
# 
#     # for lstm encoder
#     self.hidden_size = 200
#     self.num_layers = 2
#     self.lstm_dropout = 0.5
# 
#     # for feed forward
#     self.fc_hiddens_sizes = [256, 128]
#     self.dropout = 0.5
# 
#     # for training
#     self.batch_size = 10
#     self.num_epochs = 30
#     self.patience = 10
#     self.grad_clipping = 5.0
#     self.learning_rate = 0.015
#     self.weight_decay = 1e-8
#     self.lr_rate_decay = 0.98


class ProgramArgs(argparse.Namespace):
    def __init__(self):
        super(ProgramArgs, self).__init__()
        self.cuda_device = "2"
        
        self.mode = 'attack' # 'evaluate' 'attack' 'augmentation'
        
        self.workspace = '/disks/sdb/zjiehang/frequency'
        self.data = 'conll2003'
        self.label_encoding = 'BIOUL'
        self.train_data_file = "/disks/sdb/zjiehang/frequency/data/conll2003/train.txt"
        self.dev_data_file = "/disks/sdb/zjiehang/frequency/data/conll2003/dev.txt"
        self.test_data_file = "/disks/sdb/zjiehang/frequency/data/conll2003/test.txt"

        self.embedding_type = 'wordchar' # 'wordchar' or 'elmo' or 'bert'

        # for token(word) embedding
        self.token_embedding_dim = 100
        self.embedding_file = "/disks/sdb/zjiehang/DependencyParsing/pretrained_embedding/glove/glove.6B.100d.txt"

        # for char embedding (based on cnn-char encoder)
        # see: End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF
        self.char_embedding_dim = 16
        self.num_filter = 128
        self.ngram_filter_sizes = [3]
        
        # for emlo embedding
        self.elmo_options_file = "/disks/sdb/zjiehang/frequency/pretrained_embedding/elmo/options.json"
        self.elmo_weight_file = "/disks/sdb/zjiehang/frequency/pretrained_embedding/elmo/weights.hdf5"
        
        # for bert embedding
        self.bert_file = 'bert-base-cased'
        
        # for lstm encoder
        self.hidden_size = 200
        self.num_layers = 2
        self.lstm_dropout = 0.3
        
        # for feed forward
        self.fc_hiddens_sizes = [256, 128]
        self.dropout = 0.5
        
        # for training
        self.optimizer = 'adam'
        self.batch_size = 64
        self.num_epochs = 50
        self.patience = 10
        self.grad_clipping = 1.0
        self.learning_rate = 2e-5
        self.weight_decay = 0.0
        self.lr_rate_decay = None
        
        # for attack
        self.attack_method = 'brute'
        self.attack_rate = 0.2
        self.min_sentence_length = 5
        # for brute attack
        self.attack_number = 512
        
        # for defense
        self.defense = None # or 'bert'(word-level) 'spanbert'(sprase-level)
        
    @property
    def cache_path(self):
        return "{}/cache/{}".format(self.workspace, self.data)
    
    @property
    def save_path(self):
        return "{}/saved_models/{}/{}".format(self.workspace,self.data,self.embedding_type)

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
