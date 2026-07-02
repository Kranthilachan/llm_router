"""Unit tests for the data-driven business-domain classifier."""

import pytest

from app.core.domain_classifier import (
    LEARNED_DOMAIN_CONFIDENCE_THRESHOLD,
    available_domains,
    classify_domain,
)


def test_dataset_loaded_and_pipeline_trained():
    labels = set(available_domains())
    assert {"crm", "hrm", "project", "accounts"}.issubset(labels)


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("Which deals are stuck in the negotiation stage of our sales pipeline?", "crm"),
        ("Draft an onboarding plan and payroll setup for a new engineering hire.", "hrm"),
        ("Which sprint tasks are blocking the milestone next Friday?", "project"),
        ("Reconcile these AP invoices against the ledger and post journal entries.", "accounts"),
    ],
)
def test_classifier_labels_representative_prompt(prompt, expected):
    label, confidence = classify_domain(prompt)
    assert label == expected, f"Expected {expected}, got {label} (conf={confidence:.2f})"
    assert confidence >= LEARNED_DOMAIN_CONFIDENCE_THRESHOLD, (
        f"Confidence {confidence:.2f} below threshold for {expected}"
    )


def test_empty_prompt_returns_none():
    label, confidence = classify_domain("")
    assert label is None
    assert confidence == 0.0
