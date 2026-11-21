"""Hand-checkable tests for the Gate 1 pass rule (src/gate1.check_pass_rule).

Run with: python -m pytest tests/test_gate1_rule.py -v
"""
from src.gate1 import PAPER_TABLE_S4, check_pass_rule


def fake_metrics(acc0: dict, gain: dict, ci_low: dict) -> dict:
    """Minimal sigma_metrics-like dicts from percentages."""
    return {s: {"top1": [acc0[s] / 100], "gain_T": {"gain": gain[s] / 100, "ci95": [ci_low[s] / 100, 1.0]}}
            for s in acc0}


PAPER_ACC0 = {s: v[0] for s, v in PAPER_TABLE_S4.items()}
PAPER_GAIN = {s: v[1] - v[0] for s, v in PAPER_TABLE_S4.items()}   # -0.16, 2.98, 1.77, 1.46
POSITIVE_LOW = {s: 0.5 for s in PAPER_ACC0}


def test_paper_numbers_pass() -> None:
    result = check_pass_rule(fake_metrics(PAPER_ACC0, PAPER_GAIN, POSITIVE_LOW))
    assert result["pass"] and all(r["pass"] for r in result["rules"].values())


def test_rule_a_t0_off_by_more_than_3pp() -> None:
    acc0 = {**PAPER_ACC0, 1.0: PAPER_ACC0[1.0] + 3.1}
    result = check_pass_rule(fake_metrics(acc0, PAPER_GAIN, POSITIVE_LOW))
    assert not result["rules"]["a"]["pass"] and not result["pass"]
    assert result["rules"]["a"]["detail"]["1.00"] == 3.1


def test_rule_b_bounds() -> None:
    # paper gain at 0.5 is 2.98 -> allowed [1.49, 5.96]
    for g, ok in [(1.48, False), (1.50, True), (5.96, True), (6.00, False)]:
        result = check_pass_rule(fake_metrics(PAPER_ACC0, {**PAPER_GAIN, 0.5: g}, POSITIVE_LOW))
        assert result["rules"]["b"]["pass"] is ok, g
    low_ci = {**POSITIVE_LOW, 0.5: 0.0}   # CI touching 0 fails
    assert not check_pass_rule(fake_metrics(PAPER_ACC0, PAPER_GAIN, low_ci))["rules"]["b"]["pass"]


def test_rules_c_and_d() -> None:
    assert not check_pass_rule(fake_metrics(PAPER_ACC0, {**PAPER_GAIN, 1.0: 0.0}, POSITIVE_LOW))["rules"]["c"]["pass"]
    assert not check_pass_rule(fake_metrics(PAPER_ACC0, {**PAPER_GAIN, 0.0: -1.2}, POSITIVE_LOW))["rules"]["d"]["pass"]
    assert check_pass_rule(fake_metrics(PAPER_ACC0, {**PAPER_GAIN, 0.0: 1.0}, POSITIVE_LOW))["rules"]["d"]["pass"]
