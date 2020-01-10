import torch
import torch.nn.functional as F
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy

from allennlpx.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder, PretrainedBertModel
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder

from allennlp.training.optimizers import BertAdam
from torch.optim import AdamW

class BertPooler(torch.nn.Module):
    def forward(self, bert_outputs):
        return bert_outputs[:, 0]


class BertClassifier(Model):
    def __init__(self, vocab, finetunable):
        super().__init__(vocab)
        """
            A note for the pipline:
            - BertyTSVReader will use BertTokenizer to tokenize the sentence and
                generate instances, then it set the PretrainedBertIndexer to be 
                the indexer of the textfield (but it do not start the indexing process)
            - When an iterator starts iterating the dataset, it use different indexers
                to index different fields. For the `berty_tokens` field, it uses a 
                PretrainedBertIndexer to convert it into a dict containing four elements:
                    {x: ~, x-offsets: ~, x-type-ids: ~, mask: ~}    (x = berty_tokens)
            - After each field is indexed, the BasicTextFieldEmbedder will use the 
                corresponding embedders to embed them.  Here, PretrainedBertEmbedder will
                generate the embedding by accepting three args:
                    {input_ids: ~, offsets: ~, token_type_ids: ~}
                    DO NOT PASS OFFSETS INTO BERT!
                Therefore we should first keep a mapping between the input of an embedder
                and the output of a indexer.
        """
        # bert_embedder = PretrainedBertEmbedder('bert-base-uncased',
        #                                        requires_grad=finetunable,
        #                                        top_layer_only=True)
        # self.word_embedders = BasicTextFieldEmbedder(
        #     token_embedders={"berty_tokens": bert_embedder},
        #     embedder_to_indexer_map={
        #         "berty_tokens": {
        #             'input_ids': 'berty_tokens',
        #             # 'offsets': 'berty_tokens-offsets',
        #             'token_type_ids': 'berty_tokens-type-ids'
        #         }
        #     },
        #     allow_unmatched_keys=True
        #     )
        # self.pooler = BertPooler()

        self.bert_model = PretrainedBertModel.load('bert-base-uncased')
        for param in self.bert_model.parameters():
            param.requires_grad = finetunable

        self.linear = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(in_features=768, out_features=vocab.get_vocab_size('label')),
        )

        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, berty_tokens, label=None):
        # embeddings = self.word_embedders(berty_tokens)
        # encoder_out = self.pooler(embeddings)

        input_ids = berty_tokens['berty_tokens']
        token_type_ids = berty_tokens['berty_tokens-type-ids']
        _, encoder_out = self.bert_model(input_ids=input_ids,
                                         token_type_ids=token_type_ids,
                                         attention_mask=(input_ids != 0).long(),
                                         output_all_encoded_layers=False)

        logits = self.linear(encoder_out)
        output = {"logits": logits, "probs": F.softmax(logits, dim=1)}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}

    def get_optimizer(self, total_steps):
        return BertAdam(self.parameters(),
                        lr=3e-5,
                        warmup=0.1,
                        t_total=total_steps,
                        weight_decay=0.01)
        # return AdamW(self.parameters(), lr=2e-5, weight_decay=0.01)


from luna import ram_globalize


@ram_globalize()
def noise(tsr: torch.Tensor, scale=1.0):
    return tsr
    # if scale == 0:
    #     return tsr
    # else:
    #     return tsr + torch.normal(0., tsr.std().item() * scale, tsr.size(), device=tsr.device)
