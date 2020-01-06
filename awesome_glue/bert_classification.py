import torch
import torch.nn.functional as F
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy

from allennlpx.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder


class BertPooler(torch.nn.Module):
    def forward(self, bert_outputs):
        return bert_outputs[:, 0]


class BertClassifier(Model):
    def __init__(self, vocab, finetunable):
        super().__init__(vocab)

        self.word_embedders = BasicTextFieldEmbedder(
            {
                "tokens":
                PretrainedBertEmbedder(
                    'bert-base-uncased', requires_grad=finetunable, top_layer_only=True)
            },
            allow_unmatched_keys=True)

        self.pooler = BertPooler()

        self.linear = torch.nn.Linear(in_features=768, out_features=vocab.get_vocab_size('label'))

        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, tokens, label=None):
        # mask = get_text_field_mask(tokens)
        embeddings = self.word_embedders(tokens)
        encoder_out = self.pooler(embeddings)
        logits = self.linear(encoder_out)
        output = {"logits": logits, "probs": F.softmax(logits, dim=1)}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}
