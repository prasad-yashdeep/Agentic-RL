from sarl.protocol import SUBMIT, format_observation, is_submit, parse_reply, tool_call_text


def test_hermes_and_diagram_forms():
    assert parse_reply(tool_call_text("ls -la", "look")).command == "ls -la"
    assert parse_reply('<tool_call>{"name":"bash","command":"pwd"}</tool_call>').command == "pwd"  # the diagram's shorthand
    assert parse_reply('<tool_call>{"name": "bash", "arguments": "{\\"command\\": \\"id\\"}"}</tool_call>').command == "id"


def test_format_errors_do_not_crash():
    assert not parse_reply("I will run ls").ok
    r = parse_reply(tool_call_text("ls") + tool_call_text("pwd"))
    assert not r.ok and r.n_calls == 2 and "exactly ONE" in r.error
    assert "not valid JSON" in parse_reply("<tool_call>{bad json}</tool_call>").error
    # a call that decodes to a string, not an object, must not raise (the lecturer's asyncio.gather crash)
    assert "decoded to a str" in parse_reply('<tool_call>"ls"</tool_call>').error
    assert "unclosed" in parse_reply('<tool_call>{"name":"bash","arguments":{"command":"ls"}').error
    assert "unknown tool" in parse_reply('<tool_call>{"name":"python","arguments":{"command":"1"}}</tool_call>').error
    assert "no non-empty" in parse_reply('<tool_call>{"name":"bash","arguments":{"command":"  "}}</tool_call>').error


def test_submit_and_observation():
    assert is_submit(f"echo {SUBMIT}") and is_submit(f"echo '{SUBMIT}'") and not is_submit("echo done")
    o = format_observation(0, "hello\n")
    assert o.startswith("<returncode>0</returncode>") and "<output>\nhello\n</output>" in o
    big = format_observation(1, "x" * 25_000, max_chars=10_000)
    assert "<warning>" in big and "<output_head>" in big and "<output_tail>" in big and len(big) < 11_000
    assert format_observation(None, "", timed_out=True).startswith("<returncode>timeout</returncode>")
