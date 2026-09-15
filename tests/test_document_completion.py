import numpy as np
import pandas as pd
from scipy import sparse

from src.v6.document_completion import make_document_completion


def test_completion_is_exact_nonempty_and_repeatable():
    counts = sparse.csr_matrix(
        np.asarray([[2, 1, 0], [0, 3, 2], [1, 0, 0]], dtype=np.int64)
    )
    rows = pd.DataFrame(
        {
            "post_id": ["a", "b", "short"],
            "city": ["beijing", "shanghai", "xiamen"],
            "split": ["train"] * 3,
        }
    )
    first = make_document_completion(
        counts, rows, seed=41, observed_fraction=0.5, minimum_total_tokens=2
    )
    second = make_document_completion(
        counts, rows, seed=41, observed_fraction=0.5, minimum_total_tokens=2
    )
    assert first.report["maximum_reconstruction_error"] == 0
    assert first.report["all_retained_sides_nonempty"]
    assert first.retained_rows["post_id"].tolist() == ["a", "b"]
    assert first.excluded_rows["post_id"].tolist() == ["short"]
    assert (first.observed != second.observed).nnz == 0
    assert (first.target != second.target).nnz == 0


def test_completion_is_post_id_stable_under_row_permutation():
    counts = sparse.csr_matrix(
        np.asarray([[4, 2, 1], [1, 3, 2]], dtype=np.int64)
    )
    rows = pd.DataFrame(
        {
            "post_id": ["first", "second"],
            "city": ["beijing", "shanghai"],
            "split": ["validation", "validation"],
        }
    )
    normal = make_document_completion(
        counts, rows, seed=77, observed_fraction=0.5
    )
    reversed_result = make_document_completion(
        counts[[1, 0]], rows.iloc[[1, 0]].reset_index(drop=True), seed=77, observed_fraction=0.5
    )
    normal_by_id = {
        post_id: normal.observed.getrow(index).toarray()
        for index, post_id in enumerate(normal.retained_rows["post_id"])
    }
    reversed_by_id = {
        post_id: reversed_result.observed.getrow(index).toarray()
        for index, post_id in enumerate(reversed_result.retained_rows["post_id"])
    }
    assert normal_by_id.keys() == reversed_by_id.keys()
    assert all(
        np.array_equal(normal_by_id[post_id], reversed_by_id[post_id])
        for post_id in normal_by_id
    )
