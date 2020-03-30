import datetime
import logging
import math
import os
import re
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any

try:
    from apex import amp
except ImportError:
    amp = None
import torch
import torch.distributed as dist
import torch.optim.lr_scheduler
from overrides import overrides
from torch.nn.parallel import DistributedDataParallel


from allennlp.common import Lazy, Tqdm
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common import util as common_util

from allennlp.data import DataLoader

from allennlp.data.dataloader import TensorDict
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter
from allennlp.training.trainer import Trainer
from allennlpx.training.util import evaluate

logger = logging.getLogger(__name__)
class MyTrainer(Trainer):
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        data_loader: torch.utils.data.DataLoader,
        patience: Optional[int] = None,
        validation_metric: str = "-loss",
        validation_data_loader: torch.utils.data.DataLoader = None,
        test_data_loader: torch.utils.data.DataLoader = None,
        num_epochs: int = 20,
        serialization_dir: Optional[str] = None,
        num_serialized_models_to_keep: int = 20,
        keep_serialized_model_every_num_seconds: int = None,
        checkpointer: Checkpointer = None,
        model_save_interval: float = None,
        cuda_device: int = -1,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        momentum_scheduler: Optional[MomentumScheduler] = None,
        summary_interval: int = 100,
        histogram_interval: int = None,
        should_log_parameter_statistics: bool = True,
        should_log_learning_rate: bool = False,
        log_batch_size_period: Optional[int] = None,
        moving_average: Optional[MovingAverage] = None,
        distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        opt_level: Optional[str] = None,
    ) -> None:
        """
        A trainer for doing supervised learning. It just takes a labeled dataset
        and a `DataLoader`, and uses the supplied `Optimizer` to learn the weights
        for your model over some fixed number of epochs. You can also pass in a validation
        dataloader and enable early stopping. There are many other bells and whistles as well.

        # Parameters

        model : `Model`, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their `forward` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.

            If you are training your model using GPUs, your model should already be
            on the correct device. (If you use `Trainer.from_params` this will be
            handled for you.)
        optimizer : `torch.nn.Optimizer`, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        data_loader : `DataLoader`, required.
            A pytorch `DataLoader` containing your `Dataset`, yielding padded indexed batches.
        patience : Optional[int] > 0, optional (default=None)
            Number of epochs to be patient before early stopping: the training is stopped
            after `patience` epochs with no improvement. If given, it must be `> 0`.
            If None, early stopping is disabled.
        validation_metric : str, optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an `is_best` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        validation_dataloader : `DataLoader`, optional (default=None)
            A `DataLoader` to use for the validation set.  If `None`, then
            use the training `DataLoader` with the validation data.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_dir : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        num_serialized_models_to_keep : `int`, optional (default=20)
            Number of previous model checkpoints to retain.  Default is to keep 20 checkpoints.
            A value of None or -1 means all checkpoints will be kept.
        keep_serialized_model_every_num_seconds : `int`, optional (default=None)
            If num_serialized_models_to_keep is not None, then occasionally it's useful to
            save models at a given interval in addition to the last num_serialized_models_to_keep.
            To do so, specify keep_serialized_model_every_num_seconds as the number of seconds
            between permanently saved checkpoints.  Note that this option is only used if
            num_serialized_models_to_keep is not None, otherwise all checkpoints are kept.
        checkpointer : `Checkpointer`, optional (default=None)
            An instance of class Checkpointer to use instead of the default. If a checkpointer is specified,
            the arguments num_serialized_models_to_keep and keep_serialized_model_every_num_seconds should
            not be specified. The caller is responsible for initializing the checkpointer so that it is
            consistent with serialization_dir.
        model_save_interval : `float`, optional (default=None)
            If provided, then serialize models every `model_save_interval`
            seconds within single epochs.  In all cases, models are also saved
            at the end of every epoch if `serialization_dir` is provided.
        cuda_device : `int`, optional (default = -1)
            An integer specifying the CUDA device(s) to use for this process. If -1, the CPU is used.
            Data parallelism is controlled at the allennlp train level, so each trainer will have a single
            GPU.
        grad_norm : `float`, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : `float`, optional (default = `None`).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting `NaNs` in your gradients during training
            that are not solved by using `grad_norm`, you may need this.
        learning_rate_scheduler : `LearningRateScheduler`, optional (default = None)
            If specified, the learning rate will be decayed with respect to
            this schedule at the end of each epoch (or batch, if the scheduler implements
            the `step_batch` method). If you use `torch.optim.lr_scheduler.ReduceLROnPlateau`,
            this will use the `validation_metric` provided to determine if learning has plateaued.
            To support updating the learning rate on every batch, this can optionally implement
            `step_batch(batch_num_total)` which updates the learning rate given the batch number.
        momentum_scheduler : `MomentumScheduler`, optional (default = None)
            If specified, the momentum will be updated at the end of each batch or epoch
            according to the schedule.
        summary_interval : `int`, optional, (default = 100)
            Number of batches between logging scalars to tensorboard
        histogram_interval : `int`, optional, (default = `None`)
            If not None, then log histograms to tensorboard every `histogram_interval` batches.
            When this parameter is specified, the following additional logging is enabled:
                * Histograms of model parameters
                * The ratio of parameter update norm to parameter norm
                * Histogram of layer activations
            We log histograms of the parameters returned by
            `model.get_parameters_for_histogram_tensorboard_logging`.
            The layer activations are logged for any modules in the `Model` that have
            the attribute `should_log_activations` set to `True`.  Logging
            histograms requires a number of GPU-CPU copies during training and is typically
            slow, so we recommend logging histograms relatively infrequently.
            Note: only Modules that return tensors, tuples of tensors or dicts
            with tensors as values currently support activation logging.
        should_log_parameter_statistics : `bool`, optional, (default = True)
            Whether to send parameter statistics (mean and standard deviation
            of parameters and gradients) to tensorboard.
        should_log_learning_rate : `bool`, optional, (default = False)
            Whether to send parameter specific learning rate to tensorboard.
        log_batch_size_period : `int`, optional, (default = `None`)
            If defined, how often to log the average batch size.
        moving_average : `MovingAverage`, optional, (default = None)
            If provided, we will maintain moving averages for all parameters. During training, we
            employ a shadow variable for each parameter, which maintains the moving average. During
            evaluation, we backup the original parameters and assign the moving averages to corresponding
            parameters. Be careful that when saving the checkpoint, we will save the moving averages of
            parameters. This is necessary because we want the saved model to perform as well as the validated
            model if we load it later. But this may cause problems if you restart the training from checkpoint.
        distributed : `bool`, optional, (default = False)
            If set, PyTorch's `DistributedDataParallel` is used to train the model in multiple GPUs. This also
            requires `world_size` to be greater than 1.
        local_rank : `int`, optional, (default = 0)
            This is the unique identifier of the `Trainer` in a distributed process group. The GPU device id is
            used as the rank.
        world_size : `int`, (default = 1)
            The number of `Trainer` workers participating in the distributed training.
        num_gradient_accumulation_steps : `int`, optional, (default = 1)
            Gradients are accumulated for the given number of steps before doing an optimizer step. This can
            be useful to accommodate batches that are larger than the RAM size. Refer Thomas Wolf's
            [post](https://tinyurl.com/y5mv44fw) for details on Gradient Accumulation.
        opt_level : `str`, optional, (default = `None`)
            Each opt_level establishes a set of properties that govern Amp’s implementation of pure or mixed
            precision training. Must be a choice of `"O0"`, `"O1"`, `"O2"`, or `"O3"`.
            See the Apex [documentation](https://nvidia.github.io/apex/amp.html#opt-levels-and-properties) for
            more details. If `None`, Amp is not used. Defaults to `None`.
        """
        super().__init__(model,
                         optimizer,
                         data_loader,
                         patience,
                         validation_metric,
                         validation_data_loader,
                         num_epochs,
                         serialization_dir,
                         num_serialized_models_to_keep,
                         keep_serialized_model_every_num_seconds,
                         checkpointer,
                         model_save_interval,
                         cuda_device,
                         grad_norm,
                         grad_clipping,
                         learning_rate_scheduler,
                         momentum_scheduler,
                         summary_interval,
                         histogram_interval,
                         should_log_parameter_statistics,
                         should_log_learning_rate,
                         log_batch_size_period,
                         moving_average,
                         distributed,
                         local_rank,
                         world_size,
                         num_gradient_accumulation_steps,
                         opt_level)
        self._test_data_loader = test_data_loader


    @overrides
    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError(
                "Could not recover training from the checkpoint.  Did you mean to output to "
                "a different serialization directory or delete the existing serialization "
                "directory?"
            )

        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        metrics["best_epoch"] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value

        for epoch in range(epoch_counter, self._num_epochs):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            # get peak of memory usage
            if "cpu_memory_MB" in train_metrics:
                metrics["peak_cpu_memory_MB"] = max(
                    metrics.get("peak_cpu_memory_MB", 0), train_metrics["cpu_memory_MB"]
                )
            for key, value in train_metrics.items():
                if key.startswith("gpu_"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)

            if self._validation_data_loader is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self._validation_loss()

                    # It is safe again to wait till the validation is done. This is
                    # important to get the metrics right.
                    if self._distributed:
                        dist.barrier()

                    val_metrics = training_util.get_metrics(
                        self.model,
                        val_loss,
                        num_batches,
                        reset=True,
                        world_size=self._world_size,
                        cuda_device=[self.cuda_device],
                    )

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self._validation_metric]
                    self._metric_tracker.add_metric(this_epoch_val_metric)

                    if self._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break
            if self._test_data_loader is not None:
                evaluate(self.model, self._test_data_loader, cuda_device=0, batch_weight_key=None)

            if self._master:
                self._tensorboard.log_metrics(
                    train_metrics, val_metrics=val_metrics, log_to_console=True, epoch=epoch + 1
                )  # +1 because tensorboard doesn't like 0

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics["best_epoch"] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = val_metrics

            if self._serialization_dir and self._master:
                common_util.dump_metrics(
                    os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"), metrics
                )

            # The Scheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)
            if self._momentum_scheduler:
                self._momentum_scheduler.step(this_epoch_val_metric, epoch)

            if self._master:
                self._save_checkpoint(epoch)

            # Wait for the master to finish saving the checkpoint
            if self._distributed:
                dist.barrier()

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * (
                    (self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1
                )
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        # make sure pending events are flushed to disk and files are closed properly
        self._tensorboard.close()

        # Load the best model state before returning
        best_model_state = self._checkpointer.best_model_state()
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return metrics