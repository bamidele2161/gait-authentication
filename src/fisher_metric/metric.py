"""Small-sample regularized Fisher projection."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


class FisherMetric:
    def __init__(self, maximum_pca_components=30, output_dimensions=7, seed=42):
        self.maximum_pca_components = maximum_pca_components
        self.output_dimensions = output_dimensions
        self.seed = seed
        self.scaler = StandardScaler()
        self.pca = None
        self.lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")

    def fit(self, features, identities):
        features = np.asarray(features, dtype=np.float64)
        identities = np.asarray(identities)
        standardized = self.scaler.fit_transform(features)
        components = min(
            self.maximum_pca_components, standardized.shape[1],
            standardized.shape[0] - 1,
        )
        self.pca = PCA(n_components=components, whiten=True, random_state=self.seed)
        reduced = self.pca.fit_transform(standardized)
        self.lda.fit(reduced, identities)
        return self

    def transform(self, features):
        features = np.asarray(features, dtype=np.float64)
        reduced = self.pca.transform(self.scaler.transform(features))
        projected = self.lda.transform(reduced)[:, :self.output_dimensions]
        norms = np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-9)
        return projected / norms
