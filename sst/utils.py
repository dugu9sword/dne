import os
import torch
from allennlp.models import Model
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.training.optimizers import DenseSparseAdam
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer

def get_data_reader(token_indexer: TokenIndexer, subtree: bool = True):
    return StanfordSentimentTreeBankDatasetReader(token_indexers={"tokens": token_indexer},
                                                  granularity='2-class',
                                                  use_subtrees=subtree)

# load dataset
def get_data(sub_reader: DatasetReader, reader: DatasetReader, train: str, dev: str, test: str):
    sub_train_data = sub_reader.read(train)
    train_data = reader.read(train)
    dev_data = reader.read(dev)
    test_data = reader.read(test)
    return sub_train_data, train_data, dev_data, test_data

def get_token_indexer(pretrain: str):
    if pretrain == 'elmo':
        token_indexer = ELMoTokenCharactersIndexer()
    else:
        token_indexer = SingleIdTokenIndexer(lowercase_tokens=True)
    return token_indexer

def get_iterator(batch_size: int, type='bucket'):
    if type == 'bucket':
        return BucketIterator(batch_size=batch_size, sorting_keys=[("tokens", "num_tokens")])
    else:
        return BasicIterator(batch_size=batch_size)

def get_optimizer(model: Model, type: str = 'sparse', learning_rate: float = 1e-3, weight_decay: float = 0.0):
    if type == 'sparse':
        return DenseSparseAdam([{
            'params': model.word_embedders.parameters(),
            'lr': learning_rate * 0.1
        },{
            'params': list(model.parameters())[1:],
            'lr': learning_rate
        }])
    elif type == 'adam':
        return torch.optim.Adam(model.parameters(),lr=learning_rate)
    elif type == 'sgd':
        return torch.optim.SGD([{
                'params': model.word_embedders.parameters(),
                'lr': learning_rate * 0.1
            }, {
                'params': list(model.parameters())[1:],
                'lr': learning_rate
        }], weight_decay=weight_decay)
        
    
def get_best_checkpoint(path: str):
    # The author is lazy ,so it is the most easy way in allennlp to find the best
    assert os.path.exists(path) and os.path.isdir(path)    
    return "{}/best.th".format(path)