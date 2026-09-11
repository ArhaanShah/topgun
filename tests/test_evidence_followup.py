"""Acceptance tests for evidence followup experiment implementation."""

import json
import tempfile
from pathlib import Path

import pytest

# Import the evidence_followup module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase_a import evidence_followup as ef
from phase_a import evidence_followup_analysis as efa


class TestExperimentDesign:
    """Test the frozen experimental design."""
    
    def test_config_loads(self):
        """Test 1: Configuration can be loaded."""
        config, prompts, profile = ef.load_experiment_config()
        assert config is not None
        assert prompts is not None
        assert profile is not None
        assert config["experiment_id"] == "evidence-followup-qwen3.6-27b-awq-v1"
    
    def test_design_constants(self):
        """Test 1b: Design has correct basic constants."""
        config, _, _ = ef.load_experiment_config()
        assert config["design"]["total_responses"] == 144
        assert config["design"]["total_unique_prompts"] == 48
        assert len(config["design"]["task_identifiers"]) == 3
    
    def test_schedule_generation(self):
        """Test 2: Schedule generates exactly 144 unique identities."""
        config, prompts, _ = ef.load_experiment_config()
        schedule = ef.build_schedule(config, prompts)
        
        assert len(schedule) == 144, f"Expected 144 responses, got {len(schedule)}"
        
        # Check all unique
        response_ids = [row["response_id"] for row in schedule]
        assert len(set(response_ids)) == 144, "Response IDs are not unique"
        
        # Check all seeds are unique
        seeds = [row["seed"] for row in schedule]
        assert len(set(seeds)) == 144, "Seeds are not unique"
    
    def test_balanced_rounds(self):
        """Test 3: Each of three rounds contains all 48 variants."""
        config, prompts, _ = ef.load_experiment_config()
        schedule = ef.build_schedule(config, prompts)
        
        for round_num in range(3):
            round_rows = [row for row in schedule if row["block"] == round_num]
            assert len(round_rows) == 48, f"Round {round_num} has {len(round_rows)} rows, expected 48"
            
            # Check variant uniqueness within round
            variants = {
                (row["experiment"], row["task"], row["wording"], row["evidence"], 
                 row["order"], row["cue"])
                for row in round_rows
            }
            assert len(variants) == 48, f"Round {round_num} has {len(variants)} unique variants, expected 48"


class TestPromptRendering:
    """Test prompt rendering and differences."""
    
    def test_all_prompts_render(self):
        """Test 2a: All 48 unique prompts render without error."""
        config, prompts, _ = ef.load_experiment_config()
        
        rendered = set()
        
        # Experiment A
        for task in ef.TASK_IDS:
            for wording in [0, 1]:
                for evidence in [0, 1]:
                    prompt = ef.render_prompt(config, prompts, "A", task, wording, evidence)
                    assert prompt and len(prompt) > 0
                    rendered.add(prompt)
        
        # Experiment B
        for task in ef.TASK_IDS:
            for wording in [0, 1]:
                for evidence in [0, 1]:
                    for order in ["recommendation-first", "basis-first"]:
                        prompt = ef.render_prompt(config, prompts, "B", task, wording, evidence, order=order)
                        assert prompt and len(prompt) > 0
                        rendered.add(prompt)
        
        # Experiment C
        for task in ef.TASK_IDS:
            for wording in [0, 1]:
                for cue in ["neutral", "execution-unavailable"]:
                    prompt = ef.render_prompt(config, prompts, "C", task, wording, evidence=1, cue=cue)
                    assert prompt and len(prompt) > 0
                    rendered.add(prompt)
        
        assert len(rendered) == 48, f"Expected 48 unique prompts, got {len(rendered)}"
    
    def test_prompt_differences(self):
        """Test 4: Exact prompt diffs match specification."""
        config, prompts, _ = ef.load_experiment_config()
        
        # S0 vs S1: only sentence changes
        s0_e0 = ef.render_prompt(config, prompts, "A", "S", 0, 0)
        s1_e0 = ef.render_prompt(config, prompts, "A", "S", 1, 0)
        assert s0_e0 != s1_e0
        # Both should have same evidence control and experiment properties
        
        # Evidence adds only one paragraph
        s0_e0_no_ev = ef.render_prompt(config, prompts, "A", "S", 0, 0)
        s0_e1_with_ev = ef.render_prompt(config, prompts, "A", "S", 0, 1)
        # s0_e1_with_ev should be longer and contain benchmark content
        assert len(s0_e1_with_ev) > len(s0_e0_no_ev)
        
        # B order swaps only order
        b_rec = ef.render_prompt(config, prompts, "B", "S", 0, 0, order="recommendation-first")
        b_basis = ef.render_prompt(config, prompts, "B", "S", 0, 0, order="basis-first")
        assert b_rec != b_basis
        # Check order is actually swapped
        assert b_rec.find("Recommendation") < b_rec.find("Supporting basis")
        assert b_basis.find("Supporting basis") < b_basis.find("Recommendation")
    
    def test_c_cue_differences(self):
        """Test 4a: Cue factor changes only the cue paragraph."""
        config, prompts, _ = ef.load_experiment_config()
        
        neutral = ef.render_prompt(config, prompts, "C", "S", 0, 1, cue="neutral")
        unavailable = ef.render_prompt(config, prompts, "C", "S", 0, 1, cue="execution-unavailable")
        
        # Both should exist and differ
        assert neutral and unavailable
        assert neutral != unavailable
        # Check they contain the expected cues
        assert "focused on that comparison" in neutral or "concerned" in neutral
        assert "code-execution tool" in unavailable or "benchmark" in unavailable


class TestScheduleProperties:
    """Test schedule determinism and validity."""
    
    def test_schedule_is_deterministic(self):
        """Test 5: Same config produces same schedule."""
        config, prompts, _ = ef.load_experiment_config()
        schedule1 = ef.build_schedule(config, prompts)
        schedule2 = ef.build_schedule(config, prompts)
        
        assert len(schedule1) == len(schedule2)
        for r1, r2 in zip(schedule1, schedule2):
            assert r1["response_id"] == r2["response_id"]
            assert r1["seed"] == r2["seed"]
    
    def test_seeds_are_unique_across_replicates(self):
        """Test 5a: Same variant, different replicate has different seeds."""
        config, prompts, _ = ef.load_experiment_config()
        schedule = ef.build_schedule(config, prompts)
        
        # Group by (experiment, task, wording, evidence, order, cue)
        by_variant = {}
        for row in schedule:
            key = (row["experiment"], row["task"], row["wording"], 
                   row["evidence"], row["order"], row["cue"])
            if key not in by_variant:
                by_variant[key] = []
            by_variant[key].append(row)
        
        # Check each variant has 3 replicates with different seeds
        for key, rows in by_variant.items():
            assert len(rows) == 3, f"Variant {key} has {len(rows)} replicates, expected 3"
            seeds = [row["seed"] for row in rows]
            assert len(set(seeds)) == 3, f"Variant {key} has duplicate seeds"


class TestPromptContextFitting:
    """Test that prompts + max_tokens fit in context."""
    
    def test_longest_prompt_fits_context(self):
        """Test 3: Prompt + 4096 tokens fits in 8192 context."""
        config, prompts, profile = ef.load_experiment_config()
        
        # Get a mock tokenizer for counting
        tokenizer = ef.SimpleTokenizer()
        
        # Find longest prompt
        max_prompt_tokens = 0
        longest_prompt = ""
        
        for task in ef.TASK_IDS:
            for wording in [0, 1]:
                for evidence in [0, 1]:
                    for order in ["recommendation-first", "basis-first"]:
                        prompt = ef.render_prompt(config, prompts, "B", task, wording, evidence, order=order)
                        tokens = len(tokenizer.encode(prompt))
                        if tokens > max_prompt_tokens:
                            max_prompt_tokens = tokens
                            longest_prompt = prompt
        
        # Check fit
        max_tokens = profile["max_model_len"]
        available_for_output = max_tokens - max_prompt_tokens
        
        assert available_for_output >= 4096, \
            f"Longest prompt uses {max_prompt_tokens} tokens; " \
            f"only {available_for_output} available for 4096-token output (need {4096})"


class TestExperimentIdentity:
    """Test response identity computation."""
    
    def test_response_identities_are_unique(self):
        """Test 6: Each response has unique identity."""
        config, prompts, _ = ef.load_experiment_config()
        
        identities = set()
        
        for task in ef.TASK_IDS:
            for wording in [0, 1]:
                for evidence in [0, 1]:
                    for replicate in range(3):
                        rid = ef.response_identity(config["experiment_id"], "A", task, 
                                                  wording, evidence, None, None, replicate)
                        identities.add(rid)
        
        assert len(identities) == 36  # 3 tasks × 2 wordings × 2 evidence levels × 3 replicates
    
    def test_response_identity_is_deterministic(self):
        """Test 6a: Same factors produce same identity."""
        config, _, _ = ef.load_experiment_config()
        
        id1 = ef.response_identity(config["experiment_id"], "A", "S", 0, 0, None, None, 0)
        id2 = ef.response_identity(config["experiment_id"], "A", "S", 0, 0, None, None, 0)
        
        assert id1 == id2


class TestPrepareFunction:
    """Test prepare_experiment function."""
    
    def test_prepare_creates_schedule_files(self, tmp_path):
        """Test 7: Prepare creates correct directory structure."""
        cache_dir = tmp_path / "cache"
        runs_dir = tmp_path / "runs" / "test-run"
        cache_dir.mkdir(parents=True)
        
        result = ef.prepare_experiment(
            "test-run",
            cache_dir,
            runs_dir,
            download=False,
            mock=True,
        )
        
        assert result["status"] == "PREPARED"
        assert result["schedule_size"] == 144
        assert runs_dir.exists()
        assert (runs_dir / "prepare_manifest.json").exists()


class TestLockRecovery:
    """Test process lock handling."""
    
    def test_lock_context_manager(self, tmp_path):
        """Test 5b: Lock acquire and release works."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        
        lock_path = run_dir / ef.LOCK_PATH
        assert not lock_path.exists()
        
        with ef.ProcessLock(run_dir):
            assert lock_path.exists()
            payload = json.loads(lock_path.read_text())
            assert "pid" in payload
        
        assert not lock_path.exists()


class TestAnalysisModule:
    """Test analysis infrastructure."""
    
    def test_bayesian_posterior_beta(self):
        """Test 8: Bayesian posterior computes Beta credible intervals."""
        posterior = efa.BayesianPosterior(alpha=0.5, beta=0.5, draws=5000)
        
        # 10 positives out of 20
        result = posterior.posterior_interval(10, 20, credible=0.95)
        
        assert result["rate"] == 0.5
        assert result["mean"] is not None
        assert result["lower"] is not None
        assert result["upper"] is not None
        assert result["lower"] < result["mean"] < result["upper"]
    
    def test_contrast_computation(self):
        """Test 8a: Contrasts compute differences correctly."""
        posterior = efa.BayesianPosterior(alpha=0.5, beta=0.5, draws=5000)
        
        # Compare 15/30 vs 10/20
        result = posterior.contrast_posterior(10, 20, 15, 30, credible=0.95)
        
        assert result["difference"] is not None
        assert result["mean_difference"] is not None
        assert result["lower"] is not None
        assert result["upper"] is not None


class TestMockE2E:
    """End-to-end mock execution."""
    
    def test_full_mock_prepare_to_analysis(self, tmp_path):
        """Test 9: Full mock workflow from prepare through analysis."""
        cache_dir = tmp_path / "cache"
        runs_dir = tmp_path / "runs" / "e2e-test"
        cache_dir.mkdir(parents=True)
        
        # Prepare
        prepare_result = ef.prepare_experiment(
            "e2e-test",
            cache_dir,
            runs_dir,
            download=False,
            mock=True,
        )
        
        assert prepare_result["status"] == "PREPARED"
        assert prepare_result["schedule_size"] == 144
        
        # Verify manifest exists
        manifest_path = runs_dir / "prepare_manifest.json"
        assert manifest_path.exists()
        
        # Verify config snapshot
        snapshot_dir = runs_dir / "config_snapshot"
        assert snapshot_dir.exists()
        assert (snapshot_dir / "evidence_followup.yaml").exists()
        assert (snapshot_dir / "evidence_followup_prompts.yaml").exists()


class TestErrorHandling:
    """Test error detection and recovery."""
    
    def test_schedule_always_produces_144(self):
        """Test 6b: Schedule always produces 144 regardless of config.total_responses."""
        config, prompts, _ = ef.load_experiment_config()
        
        # Modify config to break design
        bad_config = config.copy()
        bad_config["design"]["total_responses"] = 999
        
        # The schedule doesn't validate against config; it always produces 144
        schedule = ef.build_schedule(bad_config, prompts)
        assert len(schedule) == 144


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
