import pytest

from phase_a.statistics import compute_wilson_interval, passes_reliability, qualifies_case, reliability_metrics


def test_wilson_interval_against_reference():
    low, high = compute_wilson_interval(3, 10)
    assert low == pytest.approx(0.1078, abs=1e-4)
    assert high == pytest.approx(0.6032, abs=1e-4)


def test_reliability_and_qualification_gates():
    metrics = reliability_metrics([(True, True)] * 8 + [(False, False)] * 2)
    assert passes_reliability(metrics)
    assert qualifies_case(3, 27, 30, True, False)
    assert not qualifies_case(2, 30, 30, True, False)
    assert not qualifies_case(3, 26, 30, True, False)
