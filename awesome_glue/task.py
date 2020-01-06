from awesome_glue.config import Config
from allennlpx.data.dataset_readers.berty_tsv import BertyTSVReader
from awesome_glue.task_specs import TASK_SPECS
from allennlp.data.vocabulary import Vocabulary
from torch.optim.adam import Adam
from allennlp.training.optimizers import DenseSparseAdam
from allennlp.data.iterators.bucket_iterator import BucketIterator
from allennlpx.training.callback_trainer import CallbackTrainer
from allennlpx.training.callbacks.evaluate_callback import EvaluateCallback, evaluate
from luna.pytorch import save_model
from allennlp.data.iterators.basic_iterator import BasicIterator
import torch
from luna.public import auto_create
from awesome_glue.bert_classification import BertClassifier


class Task:
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        spec = TASK_SPECS[config.task_id]

        self.reader = BertyTSVReader(sent1_col=spec['sent1_col'],
                                     sent2_col=spec['sent2_col'],
                                     label_col=spec['label_col'],
                                     skip_label_indexing=spec['skip_label_indexing'])

        def __load_data():
            train_data = self.reader.read(f'{spec["path"]}/train.tsv')
            dev_data = self.reader.read(f'{spec["path"]}/dev.tsv')
            test_data = self.reader.read(f'{spec["path"]}/test.tsv')
            vocab = Vocabulary.from_instances(train_data + dev_data + test_data)
            return train_data, dev_data, test_data, vocab

        # The cache name is {task}-{tokenizer}
        self.train_data, self.dev_data, self.test_data, self.vocab = auto_create(
            f"{config.task_id}-BERT", __load_data, True)

        self.model = BertClassifier(self.vocab, config.finetunable).cuda()

    def train(self):
        # yapf: disable
        if self.config.pretrain == 'bert':
            optimizer = Adam(self.model.parameters(), lr=3e-5)
        else:
            optimizer = DenseSparseAdam([
                {'params': self.model.word_embedders.parameters(), 'lr': 1e-4},
                {'params': list(self.model.parameters())[1:], 'lr': 1e-3
            }])
        # yapf: enable

        iterator = BucketIterator(batch_size=16, sorting_keys=[("tokens", "num_tokens")])
        iterator.index_with(self.vocab)

        # results for test set is not available
        dev_size = len(self.dev_data)
        tmp_valid_data = self.dev_data[:dev_size // 2]
        tmp_test_data = self.dev_data[dev_size // 2:]

        trainer = CallbackTrainer(model=self.model,
                                  optimizer=optimizer,
                                  iterator=iterator,
                                  train_dataset=self.train_data,
                                  validation_dataset=tmp_valid_data,
                                  num_epochs=4 if self.config.pretrain == 'bert' else 8,
                                  shuffle=True,
                                  patience=None,
                                  cuda_device=0,
                                  callbacks=[EvaluateCallback(tmp_test_data)])
        trainer.train()
        # log(evaluate(model, test_data, iterator, 0, None))
        # exit()
        # save_model(self.model, self.model_path)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        iterator = BasicIterator(batch_size=32)
        iterator.index_with(self.vocab)
        evaluate(self.model, self.test_data, iterator, 0, None)

    def attack(self):
        raise NotImplementedError
