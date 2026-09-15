"""Official CombinedTM adapter using the project's frozen count/embedding evidence."""

from __future__ import annotations

import random

import numpy as np

from src.v6.comparators.base import BaselineAdapter, BaselineData, BaselineFit, normalize_rows


class CombinedTMAdapter(BaselineAdapter):
    method_id = "combinedtm"
    display_name = "CombinedTM"
    dependency = "contextualized-topic-models"

    def fit(self, data: BaselineData, *, topics: int, seed: int) -> BaselineFit:
        try:
            import torch
            from contextualized_topic_models.datasets.dataset import CTMDataset
            from contextualized_topic_models.models.ctm import CombinedTM
        except ImportError as exc:  # pragma: no cover - optional environment
            raise ModuleNotFoundError(
                "CombinedTM requires the pinned optional baseline environment"
            ) from exc

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        idx2token = {index: token for index, token in enumerate(data.vocabulary)}
        train_dataset = CTMDataset(
            X_contextual=np.asarray(data.train_embeddings, dtype=np.float32),
            X_bow=data.train_counts.astype(np.float32),
            idx2token=idx2token,
            labels=None,
        )
        validation_dataset = CTMDataset(
            X_contextual=np.asarray(data.validation_embeddings, dtype=np.float32),
            X_bow=data.validation_counts.astype(np.float32),
            idx2token=idx2token,
            labels=None,
        )
        epochs = int(self.config["epochs"])
        model = CombinedTM(
            bow_size=data.vocabulary_size,
            contextual_size=int(data.train_embeddings.shape[1]),
            n_components=topics,
            model_type=str(self.config.get("model_type", "prodLDA")),
            hidden_sizes=tuple(int(value) for value in self.config.get("hidden_sizes", [100, 100])),
            activation=str(self.config.get("activation", "softplus")),
            dropout=float(self.config.get("dropout", 0.2)),
            learn_priors=bool(self.config.get("learn_priors", True)),
            batch_size=int(self.config.get("batch_size", 64)),
            lr=float(self.config.get("learning_rate", 0.002)),
            momentum=float(self.config.get("momentum", 0.99)),
            solver="adam",
            num_epochs=epochs,
            reduce_on_plateau=False,
            num_data_loader_workers=0,
        )
        # The validation partition is never passed to fit: Step 11 forbids
        # baseline-specific validation early stopping or hyperparameter choice.
        model.fit(
            train_dataset,
            validation_dataset=None,
            verbose=False,
            n_samples=int(self.config.get("theta_samples", 20)),
            do_train_predictions=False,
        )
        topic_word = normalize_rows(model.get_topic_word_distribution())
        validation_theta = normalize_rows(
            model.get_doc_topic_distribution(
                validation_dataset,
                n_samples=int(self.config.get("theta_samples", 20)),
            )
        )
        if str(self.config.get("model_type", "prodLDA")) != "prodLDA":
            raise ValueError("The frozen CombinedTM comparison requires prodLDA")
        # CombinedTM/prodLDA does not decode as theta @ softmax(beta).  Its
        # official training likelihood applies the learned raw beta logits,
        # the fitted beta BatchNorm, and then a vocabulary softmax.  Preserve
        # that exact predictive interface for document-completion NLL.
        network = model.model
        network.eval()
        device = next(network.parameters()).device
        with torch.no_grad():
            theta_tensor = torch.as_tensor(
                validation_theta, dtype=network.beta.dtype, device=device
            )
            validation_word_probabilities = torch.softmax(
                network.beta_batchnorm(theta_tensor @ network.beta), dim=1
            ).detach().cpu().numpy().astype(np.float64)
        if (
            validation_word_probabilities.shape
            != (data.validation_counts.shape[0], data.vocabulary_size)
            or not np.isfinite(validation_word_probabilities).all()
            or np.any(validation_word_probabilities < 0.0)
            or not np.allclose(
                validation_word_probabilities.sum(axis=1), 1.0,
                atol=1.0e-6, rtol=0.0,
            )
        ):
            raise FloatingPointError("CombinedTM returned invalid native probabilities")
        return BaselineFit(
            self.method_id,
            self.display_name,
            seed,
            topic_word,
            validation_theta,
            True,
            topics,
            iterations_completed=epochs,
            converged=None,
            metadata={
                "implementation": "official_contextualized_topic_models_CombinedTM",
                "embedding_source": "same_frozen_multilingual_e5_document_embeddings",
                "validation_used_during_fit": False,
                "model_type": str(self.config.get("model_type", "prodLDA")),
                "predictive_interface": (
                    "official_prodlda_beta_batchnorm_then_vocabulary_softmax"
                ),
            },
            validation_word_probabilities=validation_word_probabilities,
        )
