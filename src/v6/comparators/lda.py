"""Scikit-learn batch variational LDA baseline."""

from __future__ import annotations

from typing import Any

from sklearn.decomposition import LatentDirichletAllocation

from src.v6.comparators.base import BaselineAdapter, BaselineData, BaselineFit, normalize_rows


class LDAAdapter(BaselineAdapter):
    method_id = "lda"
    display_name = "LDA"
    dependency = "scikit-learn"

    def fit(self, data: BaselineData, *, topics: int, seed: int) -> BaselineFit:
        alpha = float(self.config.get("alpha", 1.0 / topics))
        eta = float(self.config.get("eta", 1.0 / topics))
        model = LatentDirichletAllocation(
            n_components=topics,
            doc_topic_prior=alpha,
            topic_word_prior=eta,
            learning_method="batch",
            max_iter=int(self.config["maximum_iterations"]),
            evaluate_every=-1,
            max_doc_update_iter=int(self.config.get("maximum_doc_iterations", 100)),
            mean_change_tol=float(self.config.get("document_tolerance", 1.0e-3)),
            random_state=seed,
            n_jobs=1,
            verbose=0,
        )
        model.fit(data.train_counts)
        topics_matrix = normalize_rows(model.components_)
        validation_theta = normalize_rows(model.transform(data.validation_counts))
        metadata: dict[str, Any] = {
            "implementation": "sklearn.decomposition.LatentDirichletAllocation",
            "learning_method": "batch_variational_bayes",
            "alpha": alpha,
            "eta": eta,
            "training_bound": float(model.bound_),
        }
        return BaselineFit(
            self.method_id,
            self.display_name,
            seed,
            topics_matrix,
            validation_theta,
            True,
            topics,
            iterations_completed=int(model.n_iter_),
            converged=bool(model.n_iter_ < int(self.config["maximum_iterations"])),
            metadata=metadata,
        )
