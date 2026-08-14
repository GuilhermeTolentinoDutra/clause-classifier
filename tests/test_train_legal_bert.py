import pandas as pd

from src.train_legal_bert import parse_args, prepare_data


def test_phase2_defaults_are_cpu_friendly():
    args = parse_args([])
    assert args.max_samples == 200
    assert args.epochs == 1
    assert args.batch_size == 4


def test_prepare_data_preserves_labels_and_stratifies():
    rows = []
    for label in ["A", "B", "C"]:
        rows.extend({"clause": f"text {label} {i}", "label": label} for i in range(10))
    train, test, mapping = prepare_data(pd.DataFrame(rows), max_samples=0)
    assert len(train) == 24
    assert len(test) == 6
    assert set(mapping) == {"A", "B", "C"}
    assert set(train["label_id"]) == {0, 1, 2}


def test_prepare_data_small_sample_keeps_two_examples_per_class():
    rows = []
    for label in ["A", "B", "C"]:
        rows.extend({"clause": f"text {label} {i}", "label": label} for i in range(5))
    train, test, _ = prepare_data(pd.DataFrame(rows), max_samples=6)
    assert len(train) == 3
    assert len(test) == 3
    assert set(train["label"]) == {"A", "B", "C"}
