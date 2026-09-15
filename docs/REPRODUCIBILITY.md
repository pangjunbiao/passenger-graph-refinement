# Reproducibility

Python 3.10 or later is required.

Repository verification:

    python -m pytest -q tests

The clean release was validated under Python 3.11.7 with:

    194 passed

Full reproduction additionally requires authorized datasets, stage-specific dependencies, prerequisite outputs, and the locked EnCOT parent implementation.

Locked EnCOT commit:

8ac3592165cc7851676be2776b314eb6e44e9388

Raw datasets, individual human responses, checkpoints, generated experiment outputs, and local third-party checkouts are intentionally excluded from this repository.
