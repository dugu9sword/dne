from dataclasses import dataclass

@dataclass
class CandidatePolicy:
    pass

@dataclass
class EmbeddingPolicy(CandidatePolicy):
    measure: str
    topk: int
    rho: int
        
@dataclass
class SynonymPolicy(CandidatePolicy):
    pass