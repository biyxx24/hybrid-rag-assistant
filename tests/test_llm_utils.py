from hybrid_rag.llm import Usage, count_message_tokens, count_tokens, extract_json, numbered_lines


def test_extract_json_finds_object_inside_prose():
    text = 'Sure! Here you go:\n```json\n{"route": "sql", "reason": "counts"}\n```'
    assert extract_json(text) == {"route": "sql", "reason": "counts"}


def test_extract_json_returns_empty_dict_on_garbage():
    assert extract_json("no json here") == {}
    assert extract_json("{not: valid}") == {}


def test_numbered_lines_handles_dots_and_parentheses():
    text = "1. First question?\n2) Second question?\n\n3. \"Quoted\"\nnot numbered"
    assert numbered_lines(text) == ["First question?", "Second question?", "Quoted"]


def test_token_counting_handles_gateway_model_names():
    assert count_tokens("Hello, Cornell!", "openai.gpt-4o") == count_tokens("Hello, Cornell!", "gpt-4o")
    assert count_tokens("Hello, Cornell!", "some-unknown-model") > 0
    messages = [{"role": "user", "content": "Hello"}]
    assert count_message_tokens(messages) > count_tokens("Hello")


def test_usage_since_snapshot():
    usage = Usage()
    usage.add(100, 10)
    before = usage.snapshot()
    usage.add(50, 5)
    delta = usage.since(before)
    assert (delta.prompt_tokens, delta.completion_tokens, delta.calls) == (50, 5, 1)
    assert usage.total_tokens == 165
