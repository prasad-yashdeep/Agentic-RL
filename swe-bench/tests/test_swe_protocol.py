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


def test_harness_filter_drops_test_files_only():
    from sarl.reward.grade import filter_patch

    src = "diff --git a/pkg/mod.py b/pkg/mod.py\n--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
    conf = "diff --git a/conftest.py b/conftest.py\nnew file mode 100644\n--- /dev/null\n+++ b/conftest.py\n@@ -0,0 +1 @@\n+import os\n"
    tst = "diff --git a/tests/test_mod.py b/tests/test_mod.py\n--- a/tests/test_mod.py\n+++ b/tests/test_mod.py\n@@ -1 +1 @@\n-assert f()\n+assert True\n"
    kept, dropped = filter_patch(src + conf + tst)
    assert kept == src and dropped == ["conftest.py", "tests/test_mod.py"]
    assert filter_patch("") == ("", [])


def test_mswea_protocol_matches_mini_swe_agent_v1():
    from sarl import prompts_mswea
    from sarl.protocol import format_observation_mswea, instance_prompt, parse_reply, system_prompt, tools_for

    ok = parse_reply("THOUGHT: look first.\n\n```bash\nls -la\n```", "mswea")
    assert ok.ok and ok.command == "ls -la"
    two = parse_reply("```bash\nls\n```\nthen\n```bash\npwd\n```", "mswea")
    assert not two.ok and two.error == prompts_mswea.FORMAT_ERROR.format(n=2)
    none = parse_reply("I would run ls.", "mswea")
    assert not none.ok and "found 0 actions" in none.error
    assert not parse_reply("```sh\nls\n```", "mswea").ok  # v1 accepts only ```bash
    assert tools_for("mswea") == [] and system_prompt("mswea") == prompts_mswea.SYSTEM
    ip = instance_prompt("Bug X", "mswea")
    assert ip.startswith("<pr_description>\nConsider the following PR description:\nBug X\n</pr_description>")
    assert "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached" in ip
    assert is_submit("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached")
    # observation text byte-for-byte as in the corpus: no newline is added before </output>
    assert format_observation_mswea(0, "a\n") == "<returncode>0</returncode>\n<output>\na\n</output>"
    assert format_observation_mswea(0, "") == "<returncode>0</returncode>\n<output>\n</output>"
    big = format_observation_mswea(1, "h" * 5000 + "m" * 3000 + "t" * 5000)
    assert big.startswith("<returncode>1</returncode>\n<warning>\n") and "</warning><output_head>\n" + "h" * 5000 + "\n</output_head>" in big
    assert "<elided_chars>\n3000 characters elided\n</elided_chars>\n<output_tail>\n" + "t" * 5000 + "\n</output_tail>" in big
    assert format_observation_mswea(None, "partial", command="sleep 99", timed_out=True).startswith(
        "The last command <command>sleep 99</command> timed out")
