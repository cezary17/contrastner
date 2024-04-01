import pytest

from setfit.dataset import KShotCounter

TEST_LABELS = ["PER", "ORG", "LOC", "MISC"]


def test_create_counter():
    counter = KShotCounter(k=3, labels=TEST_LABELS, mode="contrastive")
    assert counter.k == 3
    assert counter['PER'] == 0
    assert counter['ORG'] == 0
    assert counter['LOC'] == 0
    assert counter['MISC'] == 0


@pytest.mark.parametrize("labels, expected_counts", [
    ({"PER": 2, "ORG": 1}, {"PER": 1, "ORG": 0, "LOC": 0, "MISC": 0}),
    ({"PER": 2, "MISC": 1}, {"PER": 1, "ORG": 0, "LOC": 0, "MISC": 0}),
    ({"PER": 1, "ORG": 1, "LOC": 2, "MISC": 1}, {"PER": 0, "ORG": 0, "LOC": 1, "MISC": 0}),
])
def test_add_sentence(labels, expected_counts):
    counter = KShotCounter(k=3, labels=TEST_LABELS, mode="contrastive")
    result = counter.add_sentence(labels)
    assert result
    assert all(counter[label] == expected_counts[label] for label in TEST_LABELS)


@pytest.mark.parametrize("labels", [
    {"PER": 2, "ORG": 2},
    {"PER": 3, "ORG": 1},
    {"PER": 1, "ORG": 1, "LOC": 1, "MISC": 1},
    {"PER": 3, "LOC": 1, "MISC": 1}
])
def test_add_malformed_sentence(labels):
    counter = KShotCounter(k=3, labels=TEST_LABELS, mode="contrastive")
    result = counter.add_sentence(labels)
    assert not result
    assert all(counter[label] == 0 for label in TEST_LABELS)


def test_catch_bad_labels():
    counter = KShotCounter(k=3, labels=TEST_LABELS, mode="contrastive")
    with pytest.raises(KeyError):
        counter.add_sentence({"PER": 2, "ORG": 1, "BAD": 1})


def test_overfill_counter():
    counter = KShotCounter(k=3, labels=TEST_LABELS, mode="contrastive")
    counter.add_sentence({"PER": 2, "MISC": 1})
    counter.add_sentence({"PER": 2, "MISC": 1})
    counter.add_sentence({"PER": 2, "MISC": 1})
    assert not counter.add_sentence({"PER": 2, "MISC": 1})
    assert counter['PER'] == 3
