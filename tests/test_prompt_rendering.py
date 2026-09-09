from phase_a.inference import render_user_prompt
from phase_a.prepare import SimpleTokenizer


def test_prompt_rendering_has_one_user_turn_and_disables_thinking():
    rendered = render_user_prompt(SimpleTokenizer(), "hello world", "revision")
    assert rendered.raw_token_count == 2
    assert rendered.rendered.endswith("<|assistant|>\n")
    assert "system" not in rendered.rendered
    assert rendered.raw_hash != rendered.rendered_hash
