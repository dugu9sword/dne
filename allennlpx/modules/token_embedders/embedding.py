import logging
import warnings

import numpy
import torch
from allennlp.nn import util
from overrides import overrides

from allennlp.common import Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)
from allennlp.modules.token_embedders.embedding import EmbeddingsTextFile
from allennlp.modules.token_embedders.embedding import Embedding
from torch.nn.functional import embedding
from allennlp.modules.time_distributed import TimeDistributed
from allennlpx.training import adv_utils


class VanillaEmbedding(Embedding):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

    @overrides
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        original_size = tokens.size()
        tokens = util.combine_initial_dims(tokens)

        embedded = embedding(
            tokens,
            self.weight,
            padding_idx=self.padding_index,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        # Now (if necessary) add back in the extra dimensions.
        embedded = util.uncombine_initial_dims(embedded, original_size)

        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)
            embedded = projection(embedded)

        # if adv_utils.is_adv_mode():
        #     info = adv_utils.get_gradient_info()
        #     grad_norm = torch.norm(info.last_bw, dim=-1, keepdim=True) + 1e-6
        #     delta = info.last_bw / grad_norm
        #     embedded += info.grd_step * delta
        return embedded


def _read_embeddings_from_text_file(
        file_uri: str,
        embedding_dim: int,
        vocab: Vocabulary,
        namespace: str = "tokens") -> torch.FloatTensor:
    """
    Read pre-trained word vectors from an eventually compressed text file, possibly contained
    inside an archive with multiple files. The text file is assumed to be utf-8 encoded with
    space-separated fields: [word] [dim 1] [dim 2] ...

    Lines that contain more numerical tokens than `embedding_dim` raise a warning and are skipped.

    The remainder of the docstring is identical to `_read_pretrained_embeddings_file`.
    """
    tokens_to_keep = set(
        vocab.get_index_to_token_vocabulary(namespace).values())
    vocab_size = vocab.get_vocab_size(namespace)
    embeddings = {}

    # First we read the embeddings from the file, only keeping vectors for the words we need.
    logger.info("Reading pretrained embeddings from file")

    with EmbeddingsTextFile(file_uri) as embeddings_file:
        for line in Tqdm.tqdm(embeddings_file):
            token = line.split(" ", 1)[0]
            if token in tokens_to_keep:
                fields = line.rstrip().split(" ")
                if len(fields) - 1 != embedding_dim:
                    # Sometimes there are funny unicode parsing problems that lead to different
                    # fields lengths (e.g., a word with a unicode space character that splits
                    # into more than one column).  We skip those lines.  Note that if you have
                    # some kind of long header, this could result in all of your lines getting
                    # skipped.  It's hard to check for that here; you just have to look in the
                    # embedding_misses_file and at the model summary to make sure things look
                    # like they are supposed to.
                    logger.warning(
                        "Found line with wrong number of dimensions (expected: %d; actual: %d): %s",
                        embedding_dim,
                        len(fields) - 1,
                        line,
                    )
                    continue

                vector = numpy.asarray(fields[1:], dtype="float32")
                embeddings[token] = vector

    if not embeddings:
        raise ConfigurationError(
            "No embeddings of correct dimension found; you probably "
            "misspecified your embedding_dim parameter, or didn't "
            "pre-populate your Vocabulary")

    all_embeddings = numpy.asarray(list(embeddings.values()))
    float(numpy.mean(all_embeddings))
    float(numpy.std(all_embeddings))
    # Now we initialize the weight matrix for an embedding layer, starting with random vectors,
    # then filling in the word vectors we just read.
    logger.info("Initializing pre-trained embedding layer")
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).fill_(0.)
    num_tokens_found = 0
    index_to_token = vocab.get_index_to_token_vocabulary(namespace)
    for i in range(vocab_size):
        token = index_to_token[i]

        # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
        # so the word has a random initialization.
        if token in embeddings:
            embedding_matrix[i] = torch.FloatTensor(embeddings[token])
            num_tokens_found += 1
        else:
            logger.debug(
                "Token %s was not found in the embedding file. Initialising randomly.",
                token)

    logger.info("Pretrained embeddings were found for %d out of %d tokens",
                num_tokens_found, vocab_size)

    return embedding_matrix
