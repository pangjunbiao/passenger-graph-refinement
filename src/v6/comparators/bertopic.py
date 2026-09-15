"""Official BERTopic pipeline with frozen E5 embeddings and fixed-K clustering."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

from src.v6.comparators.base import BaselineAdapter, BaselineData, BaselineFit, normalize_rows


def build_training_vectorizer(
    documents: tuple[str, ...],
    common_vocabulary: tuple[str, ...],
) -> tuple[CountVectorizer, dict[str, int]]:
    """Learn BERTopic's vocabulary from frozen training documents only.

    Passing a fixed vocabulary to BERTopic can retain structurally empty
    columns when its class documents are vectorized.  Default c-TF-IDF then
    divides by a zero corpus frequency.  Learning the vectorizer vocabulary
    from the already-frozen count-space documents removes only such empty
    columns.  The caller maps the resulting weights back to the common
    vocabulary, so all methods retain the same evaluation space.
    """

    if not documents:
        raise ValueError("BERTopic requires nonempty training documents")
    if not common_vocabulary or len(common_vocabulary) != len(set(common_vocabulary)):
        raise ValueError("The common BERTopic vocabulary must be nonempty and unique")
    vectorizer = CountVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        min_df=1,
        dtype=np.int64,
    )
    audit_counts = vectorizer.fit_transform(list(documents))
    learned = tuple(map(str, vectorizer.get_feature_names_out()))
    common = set(map(str, common_vocabulary))
    unexpected = sorted(set(learned) - common)
    missing = sorted(common - set(learned))
    column_totals = np.asarray(audit_counts.sum(axis=0)).reshape(-1)
    if unexpected:
        raise ValueError(
            "BERTopic learned terms outside the frozen training vocabulary: "
            f"{unexpected[:5]}"
        )
    if missing:
        raise ValueError(
            "Frozen training vocabulary contains terms absent from its own "
            f"documents: {missing[:5]}"
        )
    if audit_counts.shape[1] != len(common_vocabulary) or np.any(column_totals <= 0):
        raise ValueError("BERTopic training vectorization retained an empty column")
    return vectorizer, {
        "training_documents": int(audit_counts.shape[0]),
        "learned_terms": int(audit_counts.shape[1]),
        "zero_frequency_terms": int(np.sum(column_totals <= 0)),
    }


class BERTopicAdapter(BaselineAdapter):
    method_id = "bertopic"
    display_name = "BERTopic (fixed-K)"
    dependency = "bertopic"

    def fit(self, data: BaselineData, *, topics: int, seed: int) -> BaselineFit:
        try:
            from bertopic import BERTopic
        except ImportError as exc:  # pragma: no cover - optional environment
            raise ModuleNotFoundError(
                "BERTopic requires the pinned optional baseline environment"
            ) from exc

        dimensions = min(
            int(self.config.get("reduced_dimensions", 5)),
            data.train_embeddings.shape[1],
            data.train_embeddings.shape[0] - 1,
        )
        if dimensions < 2:
            raise ValueError("BERTopic requires at least two reduced dimensions")
        common_index = {
            token: index for index, token in enumerate(data.vocabulary)
        }
        vectorizer, vectorizer_audit = build_training_vectorizer(
            data.train_documents,
            data.vocabulary,
        )
        reducer = PCA(n_components=dimensions, random_state=seed)
        clusterer = KMeans(
            n_clusters=topics,
            n_init=int(self.config.get("kmeans_starts", 10)),
            max_iter=int(self.config.get("kmeans_maximum_iterations", 300)),
            random_state=seed,
            algorithm="lloyd",
        )
        model = BERTopic(
            embedding_model=None,
            umap_model=reducer,
            hdbscan_model=clusterer,
            vectorizer_model=vectorizer,
            top_n_words=max(10, int(self.config.get("top_n_words", 10))),
            calculate_probabilities=False,
            low_memory=True,
            verbose=False,
        )
        train_assignments, _ = model.fit_transform(
            list(data.train_documents), data.train_embeddings
        )
        fitted_terms = tuple(
            map(str, model.vectorizer_model.get_feature_names_out())
        )
        unexpected_terms = sorted(set(fitted_terms) - set(common_index))
        if unexpected_terms:
            raise ValueError(
                "BERTopic produced terms outside the frozen vocabulary: "
                f"{unexpected_terms[:5]}"
            )
        if model.c_tf_idf_ is None or not np.isfinite(model.c_tf_idf_.data).all():
            raise ValueError("BERTopic produced a nonfinite c-TF-IDF representation")
        validation_assignments, _ = model.transform(
            list(data.validation_documents), data.validation_embeddings
        )
        observed_topics = sorted(set(map(int, train_assignments)))
        if observed_topics != list(range(topics)):
            raise ValueError(
                "Fixed-K BERTopic did not return the configured topic identifiers: "
                f"{observed_topics}"
            )
        topic_word = np.zeros((topics, data.vocabulary_size), dtype=np.float64)
        for topic in observed_topics:
            for word, weight in model.get_topic(topic):
                index = common_index.get(str(word))
                if index is not None and float(weight) > 0.0:
                    topic_word[topic, index] = float(weight)
        topic_word = normalize_rows(topic_word, floor=1.0e-15)
        validation_theta = np.zeros((len(validation_assignments), topics), dtype=np.float64)
        validation_theta[np.arange(len(validation_assignments)), validation_assignments] = 1.0
        return BaselineFit(
            self.method_id,
            self.display_name,
            seed,
            topic_word,
            validation_theta,
            False,
            topics,
            iterations_completed=int(clusterer.n_iter_),
            converged=True,
            metadata={
                "implementation": "official_bertopic_modular_pipeline",
                "embedding_source": "same_frozen_multilingual_e5_document_embeddings",
                "dimension_reduction": f"PCA({dimensions})",
                "clustering": "KMeans_fixed_K",
                "representation": "class_based_tfidf",
                "vectorizer_vocabulary_policy": (
                    "learn_on_frozen_training_documents_then_map_to_common_vocabulary"
                ),
                "vectorizer_training_documents": vectorizer_audit[
                    "training_documents"
                ],
                "vectorizer_learned_terms": vectorizer_audit["learned_terms"],
                "vectorizer_zero_frequency_terms": vectorizer_audit[
                    "zero_frequency_terms"
                ],
                "outlier_policy": "not_applicable_fixed_K_kmeans",
                "heldout_nll_status": "not_comparable_nonprobabilistic_cluster_model",
            },
        )
