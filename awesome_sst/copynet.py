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
from allennlpx.models.encoder_decoders.copynet_seq2seq import CopyNetSeq2Seq
from allennlp.data.iterators.basic_iterator import BasicIterator
from allennlp.data.dataset_readers.copynet_seq2seq import CopyNetDatasetReader
# from allennlpx.training.callback_trainer import CallbackTrainer

log_config("copy", "cf")

# reader = Seq2SeqDatasetReader(source_tokenizer=WordTokenizer(),
#                               target_tokenizer=WordTokenizer(),
#                               source_token_indexers={'tokens': SingleIdTokenIndexer()},
#                               target_token_indexers={'tokens': SingleIdTokenIndexer()})
reader = CopyNetDatasetReader(target_namespace="tgt",
                              source_tokenizer=WordTokenizer(),
                              target_tokenizer=WordTokenizer(),
                              source_token_indexers={'tokens': SingleIdTokenIndexer()})

data = reader.read('nogit/copydata.tsv')
train_data = data[:3]
validation_data = data[3:]

vocab = Vocabulary.from_instances(train_data)

model = CopyNetSeq2Seq(
    vocab,
    source_embedder=BasicTextFieldEmbedder(
        {"tokens": Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=300)}),
    encoder=PytorchSeq2SeqWrapper(torch.nn.LSTM(300, 300, batch_first=True)),
    attention=DotProductAttention(),
    beam_size=3,
    max_decoding_steps=20,
    target_embedding_dim=300,
    source_namespace="tokens",
    target_namespace="tgt",
)

# model = ComposedSeq2Seq(
#     vocab,
#     source_text_embedder=BasicTextFieldEmbedder({"tokens": token_embedder}),
#     encoder=PytorchSeq2SeqWrapper(torch.nn.LSTM(300, 300, batch_first=True)),
#     decoder=AutoRegressiveSeqDecoder(
#         vocab,
#         decoder_net=LstmCellDecoderNet(
#             decoding_dim=300,
#             target_embedding_dim=300,
#             #  attention=DotProductAttention(),
#             bidirectional_input=False),
#         max_decoding_steps=20,
#         beam_size=2,
#         target_embedder=token_embedder,
#         tie_output_embedding=True))
# model.cuda()

optimizer = optim.Adam(model.parameters())
# iterator = BucketIterator(batch_size=1, sorting_keys=[("source_tokens", "num_tokens")])
iterator = BasicIterator(batch_size=1)

iterator.index_with(vocab)
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    iterator=iterator,
    train_dataset=train_data,
    validation_dataset=validation_data,
    num_epochs=10,
    #   cuda_device=0
)

trainer.train()

predictor = Seq2SeqPredictor(model, reader)
# print(allenutil.as_sentence(validation_data[0]))
model.eval()
src_sent = allenutil.as_sentence(train_data[0].fields['source_tokens'])
print(src_sent)
print(predictor.predict(src_sent)['predicted_tokens'])

src_sent = allenutil.as_sentence(validation_data[0].fields['source_tokens'])
print(src_sent)
print(predictor.predict(src_sent)['predicted_tokens'])