from phase_a.generate import balanced_schedule
from phase_a.inference import derive_seed


def test_seed_determinism_and_uniqueness():
    seeds = {derive_seed("run", f"p{i}", j) for i in range(8) for j in range(10)}
    assert len(seeds) == 80
    assert derive_seed("run", "p1", 2) == derive_seed("run", "p1", 2)


def test_balanced_schedule():
    candidates = [{"pattern_id": f"p{i}"} for i in range(4)]
    schedule = balanced_schedule(candidates, 3, "run")
    for start in range(0, 12, 4):
        assert {row["pattern_id"] for row, _ in schedule[start : start + 4]} == {"p0", "p1", "p2", "p3"}
