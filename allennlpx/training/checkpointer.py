from allennlp.training.checkpointer import Checkpointer
import json
import numpy as np
from typing import Union, Dict, Any, List, Tuple

import logging
import os
import re
import shutil
import time

import torch

from allennlp.common import Registrable
from allennlp.nn import util as nn_util


class CheckpointerX(Checkpointer):
    def find_latest_best_checkpoint(self, latest_num, metric_key) -> Tuple[str, str]:
        """
        Return the location of the latest model and training state files.
        If there isn't a valid checkpoint then return None.
        """
        have_checkpoint = self._serialization_dir is not None and any(
            "model_state_epoch_" in x for x in os.listdir(self._serialization_dir)
        )

        if not have_checkpoint:
            return None

        serialization_files = os.listdir(self._serialization_dir)
        model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
        # Get the last checkpoint file.  Epochs are specified as either an
        # int (for end of epoch files) or with epoch and timestamp for
        # within epoch checkpoints, e.g. 5.2018-02-02-15-33-42
        found_epochs = [
            re.search(r"model_state_epoch_([0-9\.\-]+)\.th", x).group(1) for x in model_checkpoints
        ]
        int_epochs: Any = []
        for epoch in found_epochs:
            pieces = epoch.split(".")
            if len(pieces) == 1:
                # Just a single epoch without timestamp
                int_epochs.append([int(pieces[0]), "0"])
            else:
                # has a timestamp
                int_epochs.append([int(pieces[0]), pieces[1]])
        int_epochs = sorted(int_epochs, reverse=True)
        int_epochs = int_epochs[:latest_num]
        metrics = []
        for epoch_idx in int_epochs:
            epoch_to_load = str(epoch_idx[0])
            metric = json.load(open(os.path.join(
                self._serialization_dir, "metrics_epoch_{}.json".format(epoch_to_load)
            )))
            metrics.append(metric[metric_key])
        epoch_to_load = int_epochs[np.argmax(metrics)][0]
        print(f'Load checkpoint at {epoch_to_load}, with {metric_key}={round(np.max(metrics), 4)}')

        model_path = os.path.join(
            self._serialization_dir, "model_state_epoch_{}.th".format(epoch_to_load)
        )
        training_state_path = os.path.join(
            self._serialization_dir, "training_state_epoch_{}.th".format(epoch_to_load)
        )

        return (model_path, training_state_path)