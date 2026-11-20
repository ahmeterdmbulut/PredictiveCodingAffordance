"""Subset selection on a tiny hand-made index (3 classes x 4 images, no data needed).

val_id: 1  2  3  4  5  6  7  8  9  10 11 12
label:  0  1  2  0  1  2  0  1  2  0  1  2
"""
import json

import pytest

from src.imagenet_val import (
    ValImage, load_subset, parse_val_id, save_subset, select_calibration, select_per_class,
)


def _index() -> list[ValImage]:
    # stored in shuffled order, like the parquet shards
    ids = [7, 2, 11, 1, 5, 9, 12, 3, 8, 4, 10, 6]
    return [ValImage(i, f"ILSVRC2012_val_{i:08d}_n0.JPEG", (i - 1) % 3, "s.parquet", r)
            for r, i in enumerate(ids)]


def test_parse_val_id() -> None:
    assert parse_val_id("ILSVRC2012_val_00027555_n01824575.JPEG") == 27555
    with pytest.raises(ValueError):
        parse_val_id("dog.jpg")


def test_calibration_is_first_by_filename() -> None:
    assert [e.val_id for e in select_calibration(_index(), n=4)] == [1, 2, 3, 4]


def test_per_class_counts_and_order() -> None:
    chosen = select_per_class(_index(), per_class=2, seed=0)
    assert len(chosen) == 6
    assert sorted(e.label for e in chosen) == [0, 0, 1, 1, 2, 2]
    assert [e.val_id for e in chosen] == sorted(e.val_id for e in chosen)
    assert len({e.val_id for e in chosen}) == 6


def test_per_class_deterministic() -> None:
    a = select_per_class(_index(), 2, seed=0)
    assert a == select_per_class(list(reversed(_index())), 2, seed=0)  # input order does not matter
    seeds = {tuple(e.val_id for e in select_per_class(_index(), 2, seed=s)) for s in range(10)}
    assert len(seeds) > 1  # the seed actually changes the draw


def test_per_class_exclude() -> None:
    # class 0 = {1, 4, 7, 10}; excluding 1, 4 leaves exactly {7, 10}
    chosen = select_per_class(_index(), 2, seed=0, exclude={1, 4})
    assert {e.val_id for e in chosen if e.label == 0} == {7, 10}
    assert not {1, 4} & {e.val_id for e in chosen}


def test_per_class_too_few() -> None:
    with pytest.raises(ValueError):
        select_per_class(_index(), 2, seed=0, exclude={1, 4, 7})


def test_save_load_roundtrip(tmp_path) -> None:
    chosen = select_per_class(_index(), 2, seed=0)
    path = tmp_path / "subset.json"
    save_subset(path, chosen, rule="test", extra={"seed": 0})
    assert load_subset(path) == chosen
    meta = json.loads(path.read_text())
    assert (meta["n_images"], meta["n_classes"], meta["seed"]) == (6, 3, 0)
