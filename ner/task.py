import torch

from allennlp.common.tqdm import Tqdm
from allennlp.data.token_indexers.token_characters_indexer import TokenCharactersIndexer
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.token_indexers.pretrained_transformer_mismatched_indexer import PretrainedTransformerMismatchedIndexer
from allennlp.data.dataset_readers import Conll2003DatasetReader
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.token_embedders.token_characters_encoder import TokenCharactersEncoder
from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from allennlp.modules.token_embedders.pretrained_transformer_mismatched_embedder import PretrainedTransformerMismatchedEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import LstmSeq2SeqEncoder
from allennlp.training.optimizers import AdamOptimizer,SgdOptimizer,HuggingfaceAdamWOptimizer
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import StepLearningRateScheduler
from allennlp.data.dataloader import DataLoader
from allennlp.data.samplers.bucket_batch_sampler import BucketBatchSampler
from allennlpx.training.util import evaluate
from luna.public import auto_create

from ner.args import ProgramArgs
from ner.utils import utils
from ner.models.mycrftagger import MyCrfTagger
from ner.trainer.trainer import MyTrainer
from ner.trainer.predictor import NerPredictor
from ner.attacker.bruteforce import BruteForce
from ner.attacker.pwws import PWWS
from ner.utils.expandspanf1 import ExpandSpanBasedF1Measure

import logging
# disable logging from allennlp
logging.getLogger('allennlp').setLevel(logging.CRITICAL)
logging.getLogger('allennlpx').setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.CRITICAL)

class NERTask(object):
    def __init__(self, config: ProgramArgs):
        self.tokens_namespace = 'tokens'
        self.label_namespace = 'tags'
        token_indexers = {"tokens": SingleIdTokenIndexer(lowercase_tokens=True),
                          "token_characters": TokenCharactersIndexer(min_padding_length=3)}
    
        if config.embedding_type == 'elmo':
            token_indexers["elmo"] = ELMoTokenCharactersIndexer()
        elif config.embedding_type == 'bert':
            token_indexers = {'bert': PretrainedTransformerMismatchedIndexer(model_name=config.bert_file,
                                                                             namespace='bert')}
        self.data_reader = Conll2003DatasetReader(token_indexers, coding_scheme=config.label_encoding, label_namespace=self.label_namespace)
        
        data = auto_create("{}.data".format(config.data) if config.embedding_type == 'wordchar' else "{}.{}.data".format(config.data,config.embedding_type),
                           lambda: utils.load_data(self.data_reader, config),
                           True,
                           config.cache_path)
        self.vocab = data['vocab']
        self.train_data, self.dev_data, self.test_data = data['data']
        
        # for word embedding 
        token_embedders = auto_create("{}.token_embedder".format(config.data),
                                      lambda: Embedding(embedding_dim=config.token_embedding_dim, pretrained_file=config.embedding_file,vocab=self.vocab,trainable=True,vocab_namespace='tokens'),
                                      True,
                                      config.cache_path)
        self.token_embedders = token_embedders
        
        # for char embedding 
        char_embedding = Embedding(embedding_dim=config.char_embedding_dim, num_embeddings=self.vocab.get_vocab_size(namespace='token_characters'),vocab_namespace='token_characters')
        char_encoder = CnnEncoder(embedding_dim=config.char_embedding_dim,
                                  num_filters=config.num_filter,
                                  ngram_filter_sizes=config.ngram_filter_sizes,
                                  conv_layer_activation=torch.nn.ReLU())
        token_character_embedders = TokenCharactersEncoder(embedding=char_embedding,
                                                           encoder=char_encoder)

        if config.embedding_type == 'elmo':
            elmo_embedders = ElmoTokenEmbedder(options_file=config.elmo_options_file,
                                               weight_file=config.elmo_weight_file,
                                               do_layer_norm=False,
                                               dropout=0.0)
            text_filed_embedders = BasicTextFieldEmbedder({"tokens": token_embedders,
                                                           "token_characters":token_character_embedders,
                                                           "elmo":elmo_embedders})
        elif config.embedding_type == 'bert':
            bert_embedders = PretrainedTransformerMismatchedEmbedder(model_name=config.bert_file)
            text_filed_embedders = BasicTextFieldEmbedder({"bert":bert_embedders})
        else:
            text_filed_embedders = BasicTextFieldEmbedder({"tokens": token_embedders,
                                                           "token_characters":token_character_embedders})
        
        bilstm = LstmSeq2SeqEncoder(input_size=text_filed_embedders.get_output_dim(),
                                    hidden_size=config.hidden_size,
                                    num_layers=config.num_layers,
                                    dropout=config.lstm_dropout,
                                    bidirectional=True)
        
        self.model = MyCrfTagger(vocab=self.vocab,
                                 label_namespace=self.label_namespace,
                                 text_field_embedder=text_filed_embedders,
                                 encoder=bilstm if config.embedding_type!='bert' else None,
                                 # feedforward=FeedForward(input_dim=bilstm.get_output_dim(),num_layers=len(config.fc_hiddens_sizes),hidden_dims=config.fc_hiddens_sizes,activations=[torch.nn.ReLU(),torch.nn.ReLU()]),
                                 calculate_span_f1=True,
                                 label_encoding=config.label_encoding,
                                 verbose_metrics=True,
                                 dropout=config.dropout)
        # bert not fine tuned
        # if hasattr(text_filed_embedders, 'token_embedder_bert'):
        #     print('freeze bert embedding')
        #     for name, module in self.model.named_parameters():
        #         if name.startswith('text_filed_embedders.token_embedder_bert'):
        #             module.weight.requires_grad = False
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def run(self, config: ProgramArgs):
        getattr(self, config.mode)(config)
    
    def train(self, config: ProgramArgs):
        if config.embedding_type == 'bert':
            optimizer = HuggingfaceAdamWOptimizer(self.model.named_parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
        else:
            if config.optimizer == 'sgd':
                optimizer = SgdOptimizer(self.model.named_parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
            else:
                optimizer = AdamOptimizer(self.model.named_parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
        train_data_loader = DataLoader(
            self.train_data,
            batch_sampler=BucketBatchSampler(
                data_source=self.train_data,
                batch_size=config.batch_size
            ),
        )
        
        trainer = MyTrainer(model=self.model,
                          optimizer=optimizer,
                          data_loader=train_data_loader,
                          validation_data_loader=DataLoader(self.dev_data, batch_size=config.batch_size),
                          test_data_loader=DataLoader(self.test_data,batch_size=config.batch_size),
                          validation_metric='+f1-measure',
                          patience=config.patience,
                          cuda_device=0,
                          num_epochs=config.num_epochs,
                          num_serialized_models_to_keep=5,
                          grad_clipping=config.grad_clipping,
                          learning_rate_scheduler=StepLearningRateScheduler(optimizer,step_size=2,gamma=config.lr_rate_decay) if config.lr_rate_decay is not None else None,
                          serialization_dir=config.save_path
                          )
        trainer.train()
        self.evaluate(config)
        
    def evaluate(self, config: ProgramArgs):
        utils.load_best_checkpoint(self.model, config.save_path)
        evaluate(self.model,
                 data_loader=DataLoader(self.test_data, batch_size=len(self.test_data)),
                 cuda_device=0, 
                 batch_weight_key=None)
        
    def attack(self, config: ProgramArgs):
        with torch.no_grad():
            utils.load_best_checkpoint(self.model, config.save_path)
            self.model.eval()
            
            named_entity_vocab = auto_create("named_entity_vocab",
                                             lambda: utils.get_named_entity_vocab(self.vocab, self.train_data + self.dev_data + self.test_data),
                                             True,
                                             config.cache_path)

            predictor = NerPredictor(self.model, self.data_reader)
            if config.attack_method == 'pwws':
                attacker = PWWS(predictor, self.vocab, named_entity_vocab, config.attack_rate, config.min_sentence_length, policy='embedding', embedding=self.token_embedders)
            elif config.attack_method == 'brute':
                attacker = BruteForce(predictor, self.vocab, named_entity_vocab, config.attack_rate, attack_number=config.attack_number, min_sentence_length=config.min_sentence_length, policy='random')
            else:
                raise ("Unsupported attack method")
            
            clean_metrics = ExpandSpanBasedF1Measure(self.vocab, tag_namespace=self.label_namespace, label_encoding=config.label_encoding)
            adv_metrics = ExpandSpanBasedF1Measure(self.vocab, tag_namespace=self.label_namespace, label_encoding=config.label_encoding)
                
            tqdm_bar = Tqdm.tqdm(self.test_data)
            succeed_number = 0
            all_number = 0
            changed = 0 
            for instance in tqdm_bar:
                attack_result = attacker.attack_on_instance(instance)
                raw_tokens = [token.text for token in attack_result['raw_tokens'].fields[self.tokens_namespace]]
                raw_result = attack_result['raw_result']
                raw_tags = raw_result['tags']
                gold_tags = attack_result['gold_tags']
                if attack_result['attack_flag'] == 0:
                    print("\n(clean) tokens: {} tags: {}".format(' '.join(raw_tokens),' '.join(raw_tags)))
                    tqdm_bar.set_description("Not Attack! golds: {}".format(' '.join(gold_tags)))
                else:
                    all_number += 1
                    token_to_index_vocabulary = self.vocab.get_token_to_index_vocabulary(namespace=self.label_namespace)
                    raw_tags_idx = torch.Tensor([token_to_index_vocabulary[tag] for tag in raw_tags]).long()
                    golds_tags_idx = torch.Tensor([token_to_index_vocabulary[tag] for tag in gold_tags]).long()
                    clean_metric = utils.get_span_f1(raw_result['logits'], raw_tags_idx, golds_tags_idx, vocab=self.vocab, label_namespace=self.label_namespace)
                    clean_metrics += clean_metric
                    if attack_result['success'] == 1:
                        succeed_number += 1
                        changed += attack_result['changed']
                        adv_tokens = [token.text for token in attack_result['adv_tokens'].fields[self.tokens_namespace]]
                        adv_result = attack_result['adv_result']
                        adv_tags = adv_result['tags']
                        adv_tags_idx = torch.Tensor([token_to_index_vocabulary[tag] for tag in adv_tags]).long()
                        adv_metric = utils.get_span_f1(adv_result['logits'], adv_tags_idx, golds_tags_idx, vocab=self.vocab, label_namespace=self.label_namespace)
                        adv_metrics += adv_metric
                        print("\n(clean) tokens: {} tags: {}".format(' '.join(raw_tokens), ' '.join(raw_tags)))
                        print("(adv)  tokens: {} tags: {}".format(' '.join(adv_tokens), ' '.join(adv_tags)))
                        tqdm_bar.set_description("Succeed! golds:{}".format(' '.join(gold_tags)))
                    else:
                        adv_metrics += clean_metric
                        print("\n(clean) tokens: {} tags: {}".format(' '.join(raw_tokens), ' '.join(raw_tags)))
                        tqdm_bar.set_description("Failed! golds:{}".format(' '.join(gold_tags)))
                        
            print("sentence numbers: {}, among them, {} are satisfactory".format(len(self.test_data), all_number))
            print("Success Numbers: {}, Success Rate: {:.2f}%".format(succeed_number, succeed_number / all_number * 100))
            print("changed: {:.4f}".format(changed / succeed_number))
            print("clean metric: {}".format(clean_metrics))
            print("adv metric: {}".format(adv_metrics))