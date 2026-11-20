"""Check src/load_predify.HP_SETS against values typed in from the paper (not copied from HP_SETS).

Choksi et al. 2021, Predify, arXiv:2106.02749v2:
- Sec. 2.2: "set the values of hyperparameters to beta_n = 0.8, lambda_n = 0.1, and
  alpha_n = 0.01 for all the encoders/decoders in Equation (2)."
- App. A.7, Table S2 (PVGG16, tuned on Gaussian noise sigma = 0.5):
    n | beta | lambda | alpha
    1 | 0.2  | 0.05   | 0.01
    2 | 0.4  | 0.10   | 0.01
    3 | 0.4  | 0.10   | 0.01
    4 | 0.5  | 0.10   | 0.01
    5 | 0.6  | 0.00   | 0.01
Mapping to upstream buffer names: beta = ffm, lambda = fbm, alpha = erm.
"""
import pytest

from src.load_predify import HP_SETS

# (beta, lambda, alpha) per PCoder n = 1..5
PAPER_SEC_2_2_FIXED = [
    (0.8, 0.1, 0.01),
    (0.8, 0.1, 0.01),
    (0.8, 0.1, 0.01),
    (0.8, 0.1, 0.01),
    (0.8, 0.1, 0.01),
]
PAPER_TABLE_S2_TUNED = [
    (0.2, 0.05, 0.01),
    (0.4, 0.10, 0.01),
    (0.4, 0.10, 0.01),
    (0.5, 0.10, 0.01),
    (0.6, 0.00, 0.01),
]


def test_hp_set_names() -> None:
    assert set(HP_SETS) == {"fixed", "tuned"}


@pytest.mark.parametrize("name, paper", [("fixed", PAPER_SEC_2_2_FIXED), ("tuned", PAPER_TABLE_S2_TUNED)])
def test_hp_sets_match_paper(name: str, paper: list[tuple[float, float, float]]) -> None:
    hp = HP_SETS[name]
    for key in ("ffm", "fbm", "erm"):
        assert len(hp[key]) == len(paper), f"{name}.{key}: expected {len(paper)} PCoders"
    for n, (beta, lam, alpha) in enumerate(paper, start=1):
        got = (hp["ffm"][n - 1], hp["fbm"][n - 1], hp["erm"][n - 1])
        assert got == (beta, lam, alpha), f"{name} pcoder{n}: (ffm, fbm, erm) = {got}, paper says {(beta, lam, alpha)}"
