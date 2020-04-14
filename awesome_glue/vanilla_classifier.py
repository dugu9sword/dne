import torch
import torch.nn.functional as F
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.optimizers import DenseSparseAdam

from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper

from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlpx.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.token_embedders import TokenEmbedder
from luna import ram_pop, ram_has, ram_read, LabelSmoothingLoss


class EmbeddingDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super(EmbeddingDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            x_mask = torch.bernoulli(x.new_full(x.shape[:2], 1 - self.p))
            x *= x_mask.unsqueeze(dim=-1)
        return x


class Classifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        token_embedder: TokenEmbedder,
        arch: str,
        pool: str,
        num_labels: int,
    ):
        super().__init__(vocab)
        self.word_embedders = BasicTextFieldEmbedder({"tokens": token_embedder})

        self.word_drop = EmbeddingDropout(0.3)

        if arch == 'lstm':
            self.encoder = PytorchSeq2VecWrapper(
                torch.nn.LSTM(token_embedder.get_output_dim(),
                              hidden_size=300,
                              dropout=0.5,
                              num_layers=2,
                              batch_first=True))
        elif arch == 'cnn':
            self.encoder = CnnEncoder(
                embedding_dim=token_embedder.get_output_dim(),
                num_filters=100,
                ngram_filter_sizes=(3, ),
                pool=pool
            )
        elif arch == 'boe':
            self.encoder = BagOfEmbeddingsEncoder(
                embedding_dim=token_embedder.get_output_dim(), 
                averaged= pool == 'mean'
            )
        else:
            raise Exception()

        self.linear = torch.nn.Linear(
            in_features=self.encoder.get_output_dim(), out_features=num_labels)

        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()


#         self.loss_function = LabelSmoothingLoss(0.1)

    def get_optimizer(self):
        return DenseSparseAdam(self.named_parameters(), lr=1e-3)

    def forward(self, sent: TextFieldTensors, label=None):
        mask = get_text_field_mask(sent)
        embeddings = self.word_embedders(sent)
        embeddings = self.word_drop(embeddings)

        encoder_out = self.encoder(embeddings, mask)

        logits = self.linear(encoder_out)
        output = {"logits": logits, "probs": F.softmax(logits, dim=1)}
        if label is not None:
            self.accuracy(logits, label)
            if ram_has("dist_reg"):
                dist_reg = ram_read("dist_reg")
            else:
                dist_reg = 0
            output["loss"] = self.loss_function(logits, label) # + 0.1 * dist_reg
        return output

    def get_metrics(self, reset=False):
        metric = {
            'accuracy': self.accuracy.get_metric(reset),
        }
        if ram_has('dist_reg'):
            metric['dist_reg'] = ram_read("dist_reg").item()
        return metric
