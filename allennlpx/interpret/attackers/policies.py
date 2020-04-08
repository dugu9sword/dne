from dataclasses import dataclass
from typing import List, Dict


@dataclass
class CandidatePolicy:
    pass

@dataclass
class UnconstrainedPolicy:
    """
    Generate word candidates without constraints.
    """

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
    rho: float

    def cache_name(self):
        if self.topk is not None:
            return f"{self.measure}-topk-{self.topk}.txt"
        if self.rho is not None:
            return f"{self.measure}-rho-{self.rho}.txt"
        
@dataclass
class SynonymPolicy(CandidatePolicy):
    """
    Generate word candidates from the synonym dict.
    """

@dataclass
class SpecifiedPolicy(CandidatePolicy):
    """
    Generate word candidates from a given word list.
    """
    nbrs: Dict[str, List[str]]

