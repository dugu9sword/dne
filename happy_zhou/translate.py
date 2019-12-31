import itertools

import torch
import torch.optim as optim
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.activations import Activation
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.modules.attention import LinearAttention, BilinearAttention, DotProductAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, StackedSelfAttentionEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import Seq2SeqPredictor
from allennlp.training.trainer import Trainer
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import StanfordSentimentTreeBankDatasetReader
import hashlib
from luna import auto_create, load_var, log, log_config
import pathlib
from allennlp.modules.token_embedders import Embedding, TokenEmbedder
from allennlp.modules.token_embedders.embedding import \
    _read_pretrained_embeddings_file
from sst_model import LstmClassifier
from allennlpx import allenutil
from allennlpx.predictors.text_classifier import TextClassifierPredictor
from allennlp.modules.seq2seq_decoders import AutoRegressiveSeqDecoder, LstmCellDecoderNet
from allennlp.modules.seq2seq_encoders import StackedBidirectionalLstm
from allennlp.models import ComposedSeq2Seq
# from allennlpx.training.callback_trainer import CallbackTrainer
from allennlpx.models.encoder_decoders.copynet_seq2seq import CopyNetSeq2Seq
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.dataset_readers.copynet_seq2seq import CopyNetDatasetReader
from luna.pytorch import load_model

log_config("translate", "cf")

# reader = Seq2SeqDatasetReader(source_tokenizer=WordTokenizer(),
#                               target_tokenizer=WordTokenizer(),
#                               source_token_indexers={'tokens': SingleIdTokenIndexer()},
#                               target_token_indexers={'tokens': SingleIdTokenIndexer()})
reader = CopyNetDatasetReader(target_namespace="tgt",
                              source_tokenizer=WordTokenizer(),
                              target_tokenizer=WordTokenizer(),
                              source_token_indexers={'tokens': SingleIdTokenIndexer()})

train_data = reader.read('nogit/raw_att.train.tsv')
# validation_dataset = reader.read('tatoeba/en_zh.dev.tsv')

sst_reader = StanfordSentimentTreeBankDatasetReader(
    token_indexers={"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}, granularity='2-class')
sst_test_data = sst_reader.read('https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/test.txt')

vocab = Vocabulary.from_instances(train_data + sst_test_data)

sst_clf = LstmClassifier(vocab=load_var("sst_vocab"))
load_model(sst_clf, 'sst_model')
sst_predictor = TextClassifierPredictor(sst_clf, sst_reader)
sst_clf.eval()

# embedding_path = None
if pathlib.Path("/disks/sdb/zjiehang").exists():
    print("Code running in china.")
    # embedding_path = "/disks/sdb/zjiehang/frequency/pretrained_embedding/word2vec/GoogleNews-vectors-negative300.txt"
    embedding_path = "/disks/sdb/zjiehang/embeddings/fasttext/crawl-300d-2M.vec"
    # embedding_path = "/disks/sdb/zjiehang/embeddings/gensim_sgns_gnews/model.txt"
    # embedding_path = "/disks/sdb/zjiehang/embeddings/glove/glove.42B.300d.txt"
else:
    embedding_path = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"

if embedding_path:
    cache_embed_path = hashlib.md5((__name__ + embedding_path).encode()).hexdigest()
    weight = auto_create(
        cache_embed_path, lambda: _read_pretrained_embeddings_file(
            embedding_path, embedding_dim=300, vocab=vocab, namespace="tokens"), True)
    token_embedder = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                               embedding_dim=300,
                               weight=weight,
                               sparse=True,
                               trainable=False)
else:
    token_embedder = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=300)


model = CopyNetSeq2Seq(
    vocab,
    source_embedder=BasicTextFieldEmbedder(
        {"tokens": Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=300)}),
    encoder=PytorchSeq2SeqWrapper(torch.nn.LSTM(300, 300, batch_first=True)),
    attention=DotProductAttention(),
    beam_size=3,
    max_decoding_steps=50,
    target_embedding_dim=300,
    source_namespace="tokens",
    target_namespace="tgt",
)

# model = ComposedSeq2Seq(vocab,
#                         source_text_embedder=BasicTextFieldEmbedder({"tokens": token_embedder}),
#                         encoder=PytorchSeq2SeqWrapper(torch.nn.LSTM(300, 300, batch_first=True)),
#                         decoder=AutoRegressiveSeqDecoder(vocab,
#                                                          decoder_net=LstmCellDecoderNet(
#                                                              decoding_dim=300,
#                                                              target_embedding_dim=300,
#                                                             #  attention=DotProductAttention(),
#                                                              bidirectional_input=False),
#                                                          max_decoding_steps=50,
#                                                          beam_size=1,
#                                                          target_embedder=token_embedder,
#                                                          tie_output_embedding=True))
model.cuda()

optimizer = optim.Adam(model.parameters())
iterator = BucketIterator(batch_size=16, sorting_keys=[("source_tokens", "num_tokens")])

iterator.index_with(vocab)
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_data,
                  validation_dataset=train_data[:100],
                  num_epochs=1,
                  cuda_device=0)

for i in range(50):
    log('EPOCH {}'.format(i))
    model.train()
    trainer.train()

    if i >= 0:
        predictor = Seq2SeqPredictor(model, reader)

        total_num = 100
        succ_num = 0

        model.eval()
        for instance in itertools.islice(sst_test_data, total_num):
            src_words = allenutil.as_sentence(instance.fields['tokens'])
            src_probs = sst_predictor.predict(src_words)['probs']
            log('SOURCE:', src_words, target='f')
            log('Prediction:', src_probs, target='f')

            pred_words = allenutil.as_sentence(predictor.predict(src_words)['predicted_tokens'][0])
            pred_probs = sst_predictor.predict(pred_words)['probs']
            log("PRED", pred_words, target='f')
            log('Prediction:', pred_probs, target='f')
            log(target='f')
            if (src_probs[0] - src_probs[1]) * (pred_probs[0] - pred_probs[1]) < 0:
                succ_num += 1
        log('FLIP RATE {:.2f}'.format(succ_num / total_num * 100))
