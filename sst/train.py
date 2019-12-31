import os
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.util import prepare_environment
from allennlp.common.params import Params
from allennlpx.training.callback_trainer import CallbackTrainer
from allennlpx.training.callbacks.evaluate_callback import EvaluateCallback
from luna import (auto_create, log_config)
from sst.model import LstmClassifier
from sst.program_args import ProgramArgs
from sst.utils import (get_token_indexer, get_data_reader, get_data, get_iterator, get_optimizer)

# for logging
log_config("log", "cf")

if __name__ == '__main__':
    config = ProgramArgs.parse(True)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_device
    
    # random seed
    prepare_environment(Params({}))
    # get token indexer
    token_indexer = get_token_indexer(config.pretrain)
    
    # get data reader
    sub_reader = get_data_reader(token_indexer, subtree=True)
    reader = get_data_reader(token_indexer, subtree=False)
    
    # get dataset 
    sub_train_data, train_data, \
    dev_data, test_data = auto_create(f"{config.cache_prefix}_data", 
                                      lambda: get_data(sub_reader, reader, config.train_data_file, config.dev_data_file, config.test_data_file), 
                                      True, 
                                      config.cache_path)
    
    # get vocab
    vocab = auto_create(f"{config.cache_prefix}_vocab", 
                        lambda: Vocabulary.from_instances(sub_train_data + dev_data + test_data), 
                        True, 
                        config.cache_path)

    # get model
    model = LstmClassifier(vocab, config).cuda()
    
    # get iterator
    iterator = get_iterator(config.batch_size)
    iterator.index_with(vocab)
    
    # get optimizer 
    optimizer = get_optimizer(config.is_sparse_optimizer, config.learning_rate, model)
    
    # training 
    trainer = CallbackTrainer(model=model,
                              optimizer=optimizer,
                              iterator=iterator,
                              train_dataset=train_data,
                              validation_dataset=dev_data,
                              validation_metric='+accuracy',
                              num_epochs=8,
                              shuffle=True,
                              patience=None,
                              cuda_device=0,
                              callbacks=[EvaluateCallback(test_data)])
    trainer.train()
