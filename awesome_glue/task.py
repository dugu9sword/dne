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
from allennlp.training.learning_rate_schedulers.slanted_triangular import SlantedTriangular
from pytorch_pretrained_bert.optimization import BertAdam
from luna.logging import log


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
        num_epochs = 3
        pseudo_batch_size = 24
        accumulate_num = 2
        batch_size = pseudo_batch_size * accumulate_num
        optimizer = BertAdam(self.model.parameters(),
                             lr=3e-5,
                             warmup=0.1,
                             t_total=(len(self.train_data) // batch_size + 1) * num_epochs,
                             weight_decay=0.01)

        iterator = BucketIterator(batch_size=pseudo_batch_size,
                                  sorting_keys=[("berty_tokens", "num_tokens")])
        iterator.index_with(self.vocab)

        trainer = CallbackTrainer(
            model=self.model,
            optimizer=optimizer,
            iterator=iterator,
            should_log_learning_rate=True,
            train_dataset=self.train_data,
            validation_dataset=self.dev_data,
            num_epochs=num_epochs,
            shuffle=True,
            patience=None,
            cuda_device=0,
            num_gradient_accumulation_steps=accumulate_num,
            #   callbacks=[EvaluateCallback(self.dev_data)],
        )
        trainer.train()
        log(evaluate(self.model, self.dev_data, iterator, 0, None))
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
