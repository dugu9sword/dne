from dataclasses import dataclass
from typing import List


@dataclass
class CandidatePolicy:
    pass

@dataclass
class EmbeddingPolicy(CandidatePolicy):
    """
    Generate word candidates by word vectors.
        mearsure: euc/cos
        topk:     you know it
        rho:      { c | distance(c, given_word) < rho }
    """
    measure: str
    topk: int
    rho: int
        
@dataclass
class SynonymPolicy(CandidatePolicy):
    """
    Generate word candidates from the synonym dict.
    """
    pass

@dataclass
class SpecifiedPolicy(CandidatePolicy):
    """
    Generate word candidates from a given word list.
    """
    words: List[str]
