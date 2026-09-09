from phase_a.audit import select_audit_rows


def test_audit_sampling_is_balanced_and_includes_invalid(generation_factory, judgment_factory):
    responses, judgments = [], []
    for index in range(30):
        response = generation_factory(split="reproduction", sample_index=index, response_text=f"response {index}")
        responses.append(response)
        judgments.append(judgment_factory(response, label=index % 2 == 0, evidence=[]))
    judgments[29] = judgment_factory(responses[29], label=False, evidence=[], invalid=True)
    selected = select_audit_rows(responses, judgments)
    assert 8 <= len(selected) <= 9
    assert any(row.sample_index == 29 for row in selected)
    assert select_audit_rows(responses, judgments) == selected
