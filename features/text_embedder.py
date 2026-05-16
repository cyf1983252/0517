"""
文本嵌入模块：TF-IDF + SVD 字符级语义编码
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import config


class TextEmbedder:
    """TF-IDF 字符级文本嵌入"""

    def __init__(self):
        self.tfidf = None
        self.svd = None
        self.output_dim = None

    def fit(self, texts: list):
        tfidf_dim = config.TFIDF_MAX_FEATURES
        svd_dim = config.TFIDF_SVD_DIM
        self.tfidf = TfidfVectorizer(
            max_features=tfidf_dim, ngram_range=(2, 5),
            analyzer="char_wb", sublinear_tf=True,
        )
        tfidf_mat = self.tfidf.fit_transform(texts)
        n_comp = min(svd_dim, tfidf_mat.shape[1] - 1, tfidf_mat.shape[0] - 1)
        n_comp = max(n_comp, 32)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=config.RANDOM_SEED)
        self.svd.fit(tfidf_mat)
        self.output_dim = n_comp
        print(f"  TF-IDF+SVD fitted: {tfidf_mat.shape[1]}→{n_comp}d")

    def transform(self, texts: list) -> np.ndarray:
        return self.svd.transform(self.tfidf.transform(texts)).astype(np.float32)

    def fit_transform(self, texts: list) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)
