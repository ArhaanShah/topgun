import pytest
from pydantic import ValidationError

from phase_a.judge import build_judge_prompt, parse_judge_output


def test_strict_judge_json_parsing():
    parsed = parse_judge_output('{"label":true,"evidence":["exact"],"reason":"Observed.","invalid":false}')
    assert parsed.label is True
    with pytest.raises(ValidationError):
        parse_judge_output('{"label":true,"evidence":[],"reason":"x","invalid":false,"confidence":1}')
    with pytest.raises(ValueError):
        parse_judge_output("prose before {}")


def test_judge_prompt_is_blinded():
    prompt = build_judge_prompt("rubric", "user", "response")
    assert "published" not in prompt.lower()
    assert "candidate" not in prompt.lower()
    assert "original weirdchat" not in prompt.lower()
