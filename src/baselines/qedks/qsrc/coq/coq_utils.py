import re
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

RE_SPLIT = re.compile(r"[.!?；;]")

class BroadAnswerDetector:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        sim_thresh: float = 0.50,          
        min_sent: int   = 3              
    ) -> None:
        self._encoder   = None   # [QEDKS-ADAPTED E3] lazy, pinned revision
        self._model_name = model_name
        self.sim_thresh = sim_thresh
        self.min_sent   = min_sent

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = SentenceTransformer(self._model_name, revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
        return self._encoder

    @encoder.setter
    def encoder(self, v):
        self._encoder = v

    def width_score(self, answer: str) -> float:
        sents = [s.strip() for s in RE_SPLIT.split(answer) if len(s.strip()) > 4]
        if len(sents) < self.min_sent:
            return 1.0

        vecs = self.encoder.encode(sents, convert_to_numpy=True, normalize_embeddings=True)
        sims = cosine_similarity(vecs)
        tri = sims[np.triu_indices_from(sims, k=1)]
        avg_sim = float(tri.mean())          

        return round(1.0 - avg_sim, 3)

    def is_broad(self, answer: str, verbose: bool = False) -> bool:
        avg_sim = 1.0 - self.width_score(answer)
        if verbose:
            print(f"[SBERT-Detector] avg-sim={avg_sim:.3f} (threshold={self.sim_thresh})")
        return avg_sim < self.sim_thresh
