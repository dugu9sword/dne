import torch
import torch.nn.functional as F
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy
from transformers import AdamW

from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.seq2vec_encoders import ClsPooler
from luna import ram_globalize

class BertClassifier(Model):
    def __init__(self, vocab, num_labels):
        super().__init__(vocab)
        self.bert_embedder = PretrainedTransformerEmbedder('bert-base-uncased')
        self.pooler = ClsPooler(self.bert_embedder.get_output_dim())

        self.linear = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(in_features=768, out_features=num_labels))

        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()


    def forward(self, sent, label=None):
        bert_embeddings = self.bert_embedder(
            token_ids=sent['tokens']['token_ids'],
            type_ids=sent['tokens']['type_ids'],
            mask=sent['tokens']['mask'])
        bert_vec = self.pooler(bert_embeddings)

        logits = self.linear(bert_vec)
        output = {"logits": logits, "probs": F.softmax(logits, dim=1)}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}

    def get_optimizer(self):
        optimizer = AdamW(self.parameters(), lr=2e-5, eps=1e-8)
        # get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        # )
        return optimizer


@ram_globalize()
def noise(tsr: torch.Tensor, scale=1.0):
    return tsr
    # if scale == 0:
    #     return tsr
    # else:
    #     return tsr + torch.normal(0., tsr.std().item() * scale, tsr.size(), device=tsr.device)
