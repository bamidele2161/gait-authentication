"""Remove directions that make recording condition easy to predict."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class DomainAdversarialProjector:
    def __init__(self, components=30, removal_strength=1.0, seed=42):
        self.components = components
        self.removal_strength = removal_strength
        self.seed = seed
        self.scaler = StandardScaler()
        self.pca = None
        self.domain_basis = None

    def fit(self, features, domains):
        values = self.scaler.fit_transform(np.asarray(features, dtype=np.float64))
        count = min(self.components, values.shape[1], values.shape[0] - 1)
        self.pca = PCA(count, whiten=True, random_state=self.seed).fit(values)
        reduced = self.pca.transform(values)
        classifier = LogisticRegression(
            C=1.0, max_iter=2_000, class_weight="balanced", random_state=self.seed
        ).fit(reduced, domains)
        # The row space of the condition classifier contains the linear
        # directions that reveal condition. QR gives an orthonormal basis.
        basis, _ = np.linalg.qr(classifier.coef_.T)
        self.domain_basis = basis[:, :classifier.coef_.shape[0] - 1]
        return self

    def transform(self, features):
        values = np.asarray(features, dtype=np.float64)
        reduced = self.pca.transform(self.scaler.transform(values))
        condition_component = reduced @ self.domain_basis @ self.domain_basis.T
        return reduced - self.removal_strength * condition_component
