"""Nonnegative matrix factorization topic-extraction baseline."""

from __future__ import annotations

import warnings

from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfTransformer

from src.v6.comparators.base import BaselineAdapter, BaselineData, BaselineFit, normalize_rows


class NMFAdapter(BaselineAdapter):
    method_id = "nmf"
    display_name = "NMF"
    dependency = "scikit-learn"

    def fit(self, data: BaselineData, *, topics: int, seed: int) -> BaselineFit:
        maximum = int(self.config["maximum_iterations"])
        tfidf = TfidfTransformer(
            norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False
        )
        train_features = tfidf.fit_transform(data.train_counts)
        validation_features = tfidf.transform(data.validation_counts)
        model = NMF(
            n_components=topics,
            init=str(self.config.get("initialization", "nndsvdar")),
            solver="cd",
            beta_loss="frobenius",
            tol=float(self.config.get("tolerance", 1.0e-4)),
            max_iter=maximum,
            random_state=seed,
            alpha_W=0.0,
            alpha_H=0.0,
            l1_ratio=0.0,
            shuffle=False,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit_transform(train_features)
        topics_matrix = normalize_rows(model.components_)
        validation_theta = normalize_rows(
            model.transform(validation_features), floor=1.0e-15
        )
        hit_limit = any(issubclass(item.category, ConvergenceWarning) for item in caught)
        return BaselineFit(
            self.method_id,
            self.display_name,
            seed,
            topics_matrix,
            validation_theta,
            False,
            topics,
            iterations_completed=int(model.n_iter_),
            converged=not hit_limit and int(model.n_iter_) < maximum,
            metadata={
                "implementation": "sklearn.decomposition.NMF",
                "objective": "frobenius",
                "input_representation": "training_fitted_l2_tfidf",
                "reconstruction_error": float(model.reconstruction_err_),
                "heldout_nll_status": "not_comparable_nonprobabilistic_factorization",
            },
        )
