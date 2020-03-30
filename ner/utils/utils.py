import os
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
from torch.utils.data import Dataset
from allennlp.training.metrics.metric import Metric
from allennlp.models.model import Model
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary, DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from allennlp.common.tqdm import Tqdm

from ner.args import ProgramArgs
from ner.utils.expandspanf1 import ExpandSpanBasedF1Measure
from ner.utils.spanaccuracy import SpanBasedAccuracyMeasure

NON_NAMED_ENTITY_LABEL = 'O'

def load_data(reader: DatasetReader, config: ProgramArgs) -> Dict: 
    train_data = reader.read(config.train_data_file)
    dev_data = reader.read(config.dev_data_file)
    test_data = reader.read(config.test_data_file)
    vocab = Vocabulary.from_instances(train_data + dev_data + test_data)
    train_data.index_with(vocab)
    dev_data.index_with(vocab)
    test_data.index_with(vocab)
    
    return {"data": (train_data, dev_data, test_data), "vocab": vocab}

def get_best_checkpoint(path: str) -> str:
    # The author is lazy ,so it is the most easy way in allennlp to find the best
    assert os.path.exists(path) and os.path.isdir(path)
    best_path = "{}/best.th".format(path)
    assert os.path.exists(best_path) and os.path.isfile(best_path)
    return best_path

def load_best_checkpoint(model: Model, save_path: str):
    best_checkpoint_path = get_best_checkpoint(save_path)
    print('load model from {}'.format(best_checkpoint_path))
    model.load_state_dict(torch.load(best_checkpoint_path))

def get_named_entity_vocab(vocab: Vocabulary, 
                           data: Dataset,
                           tokens: str = 'tokens',
                           tags: str = 'tags') -> Dict[int, str]:
    named_entity_vocab = {}
    for instance in Tqdm.tqdm(data):
        for word, tag in zip(instance.fields[tokens], instance.fields[tags]):
            if tag != NON_NAMED_ENTITY_LABEL:
                if word.text not in named_entity_vocab:
                    named_entity_vocab[word.text.lower()] = vocab.get_token_index(word.text.lower(), namespace=tokens)
    named_entity_vocab[DEFAULT_PADDING_TOKEN] = vocab.get_token_index(DEFAULT_PADDING_TOKEN)
    named_entity_vocab[DEFAULT_OOV_TOKEN] = vocab.get_token_index(DEFAULT_OOV_TOKEN)
    return named_entity_vocab

# forward on instance returns numpy array
# which make the following operation error
# for 1-batch result, unsqueezing  
def to_tensor(result: Dict[str, Any])-> Dict :
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            result[key] = torch.from_numpy(value)
    return result


def get_metric(vocab: Vocabulary,
               logits: torch.Tensor, 
               predicted_tags: List, 
               tags: List, 
               mask: Optional[torch.BoolTensor] = None,
               metric_type: str ='accuracy', 
               label_encoding: str ='BIOUL') -> Metric:
    if metric_type == 'accuracy':
        return get_span_accuracy(vocab, predicted_tags, tags, mask, label_encoding)
    elif metric_type == 'f1':
        token_to_index_vocabulary = vocab.get_token_to_index_vocabulary(namespace='labels')
        predicts = torch.Tensor([token_to_index_vocabulary[tag] for tag in predicted_tags]).long()
        golds = torch.Tensor([token_to_index_vocabulary[tag] for tag in tags]).long()
        return get_span_f1(logits, predicts, golds, mask, vocab, label_encoding)
    else:
        raise ("Error type in metric !")
    
    
def get_span_accuracy(vocab: Vocabulary, 
                      predicts: List, 
                      golds: List, 
                      mask: Optional[torch.BoolTensor] = None,
                      label_encoding: str = 'BIOUL') -> Metric:
    metric = SpanBasedAccuracyMeasure(vocab, tag_namespace='labels', label_encoding=label_encoding)
    metric(predicts, golds, mask)
    return metric

def get_span_f1(logits: Union[torch.Tensor, np.ndarray], 
                predictions: Union[torch.Tensor, np.ndarray], 
                tags: Union[torch.Tensor, np.ndarray], 
                vocab: Vocabulary,
                mask: Optional[torch.BoolTensor] = None,
                label_encoding: str ='BIOUL',
                label_namespace: str = 'tags'):
    logits, predictions, tags = map(lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x, [logits, predictions, tags]) 
    metric = ExpandSpanBasedF1Measure(vocab, tag_namespace=label_namespace, label_encoding=label_encoding)

    class_probabilities = logits * 0.0
    for i, tag in enumerate(predictions):
        class_probabilities[i, tag] = 1

    # unsqueezing the 1-batch to [batch_size (1) , seq len , classes]
    if len(class_probabilities.shape) == 2:
        metric(class_probabilities.unsqueeze(0), tags.unsqueeze(0), mask.unsqueeze(0) if mask is not None else None)
    else:
        metric(class_probabilities, tags, mask)
    return metric