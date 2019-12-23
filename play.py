# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'

import torch

from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import StanfordSentimentTreeBankDatasetReader
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_characters_indexer import TokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder, TextFieldEmbedder
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
from allennlp.modules.token_embedders import Embedding
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, Seq2VecEncoder
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.nn.util import get_text_field_mask
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
import pathlib
import hashlib
from allennlpx.interpret.pgd import PGD, DEFAULT_IGNORE_TOKENS
from allennlpx.interpret.hotflip import HotFlip
from allennlpx.predictors.predictor import PredictorX
from allennlpx import allenutil
from luna import auto_create, flt2str

train_reader = StanfordSentimentTreeBankDatasetReader(
    token_indexers={"tokens": SingleIdTokenIndexer(lowercase_tokens=True)},
    granularity='2-class',
    use_subtrees=True)
reader = StanfordSentimentTreeBankDatasetReader(
    token_indexers={"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}, granularity='2-class')


def load_data():
    train_data = train_reader.read(
        'https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt')
    dev_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt')
    test_data = reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/test.txt')
    return train_data, dev_data, test_data


train_data, dev_data, test_data = auto_create("sst", load_data, True)

vocab = Vocabulary.from_instances(train_data)

# embedding_path = None
# embedding_path = "/disks/sdb/zjiehang/embeddings/fasttext/crawl-300d-2M.vec"
embedding_path = "/disks/sdb/zjiehang/embeddings/gensim_sgns_gnews/model.txt"
# embedding_path = "/disks/sdb/zjiehang/embeddings/glove/glove.42B.300d.txt"

if embedding_path:
    cache_embed_path = hashlib.md5(embedding_path.encode()).hexdigest()
    weight = auto_create(
        cache_embed_path, lambda: _read_pretrained_embeddings_file(
            embedding_path, embedding_dim=300, vocab=vocab, namespace="tokens"), True)
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=300,
                                weight=weight,
                                trainable=False)

else:
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=300)

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
encoder = PytorchSeq2VecWrapper(
    torch.nn.LSTM(
        300,
        hidden_size=512,
        num_layers=2,
        #                                               bidirectional=True,
        batch_first=True))


class LstmClassifier(Model):
    def __init__(self, word_embeddings, encoder, vocab):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=encoder.get_output_dim(),
                            out_features=vocab.get_vocab_size('label')))
        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, tokens, label=None):
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)
        #         print(encoder_out.size(), logits.size())
        output = {"logits": logits, "probs": F.softmax(logits, dim=1)}
        #         print(output)
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}


model = LstmClassifier(word_embeddings, encoder, vocab)
model.cuda()

model_path = 'sst_model.pt'
if pathlib.Path(model_path).exists():
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
else:
    iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
    # iterator = BasicIterator(batch_size=32)
    iterator.index_with(vocab)
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-5)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_data,
        validation_dataset=dev_data,
        #                   test_dataset=test_data,
        num_epochs=5,
        shuffle=True,
        patience=5,
        cuda_device=0)
    trainer.train()
    with open(model_path, 'wb') as f:
        torch.save(model.state_dict(), f)

for module in model.modules():
    if isinstance(module, TextFieldEmbedder):
        for embed in module._token_embedders.keys():
            module._token_embedders[embed].weight.requires_grad = True

pos_words = [line.rstrip('\n') for line in open("sentiment-words/positive-words.txt")]
neg_words = [line.rstrip('\n') for line in open("sentiment-words/negative-words.txt")]
not_words = [line.rstrip('\n') for line in open("sentiment-words/negation-words.txt")]
forbidden_words = pos_words + neg_words + not_words + DEFAULT_IGNORE_TOKENS

from allennlp.interpret.attackers.hotflip import Hotflip
from allennlpx.predictors.text_classifier import TextClassifierPredictorX

predictor = TextClassifierPredictorX(model.cpu(), reader)
# attacker = HotFlip(predictor)
attacker = PGD(predictor)
attacker.initialize()

for i in range(10):
    raw_text = allenutil.as_sentence(test_data[i])
    print(raw_text)
    print("\t", flt2str(predictor.predict(raw_text)['probs']))

    result = attacker.attack_from_json({"sentence": raw_text},
                                       ignore_tokens=forbidden_words,
                                       forbidden_tokens=forbidden_words,
                                       step_size=100,
                                       max_change_num=3,
                                       iter_change_num=2)
    att_text = allenutil.as_sentence(result['att'])
    print(att_text)
    print('\t', flt2str(predictor.predict(att_text)['probs']))
    print()
