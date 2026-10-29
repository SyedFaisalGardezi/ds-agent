"""Tests for agent/api/kernel.py — SessionKernel and CellOutput."""
from __future__ import annotations

import pytest

from agent.api.kernel import CellOutput, SessionKernel, _compile_with_repr

# ── CellOutput ────────────────────────────────────────────────────────────────

def test_cell_output_is_empty_true():
    assert CellOutput().is_empty() is True


def test_cell_output_is_empty_false_stdout():
    assert CellOutput(stdout="hi").is_empty() is False


def test_cell_output_is_empty_false_error():
    assert CellOutput(error="boom").is_empty() is False


def test_cell_output_is_empty_false_figures():
    assert CellOutput(figures=["base64data"]).is_empty() is False


def test_cell_output_is_empty_false_repr():
    assert CellOutput(last_repr="42").is_empty() is False


def test_cell_output_as_text_no_output():
    assert CellOutput().as_text() == "(no output)"


def test_cell_output_as_text_stdout():
    out = CellOutput(stdout="hello\n")
    text = out.as_text()
    assert "OUTPUT:" in text
    assert "hello" in text


def test_cell_output_as_text_error():
    out = CellOutput(error="Traceback (most recent call last):\n  ...", success=False)
    text = out.as_text()
    assert "ERROR:" in text
    assert "Traceback" in text


def test_cell_output_as_text_truncates_stdout():
    long_out = "x" * 5000
    out = CellOutput(stdout=long_out)
    text = out.as_text(max_chars=100)
    assert "truncated" in text
    assert len(text) < 5000


def test_cell_output_as_text_repr_shown_when_no_stdout():
    out = CellOutput(last_repr="42")
    text = out.as_text()
    assert "RESULT:" in text
    assert "42" in text


def test_cell_output_as_text_repr_truncated():
    out = CellOutput(last_repr="x" * 2000)
    text = out.as_text()
    assert "…" in text


def test_cell_output_as_text_figures():
    out = CellOutput(figures=["base64img1", "base64img2"])
    text = out.as_text()
    assert "2 figure(s)" in text


def test_cell_output_as_text_stderr():
    out = CellOutput(stderr="Warning: deprecated")
    text = out.as_text()
    assert "STDERR:" in text


# ── SessionKernel ─────────────────────────────────────────────────────────────

@pytest.fixture
def kernel():
    return SessionKernel()


def test_kernel_boot_preloads_pandas(kernel):
    assert "pd" in kernel.namespace
    assert kernel.namespace["pd"] is not None


def test_kernel_boot_preloads_numpy(kernel):
    assert "np" in kernel.namespace


def test_kernel_boot_sets_df_none(kernel):
    assert kernel.namespace["df"] is None


def test_kernel_execute_print(kernel):
    out = kernel.execute('print("hello world")')
    assert out.success is True
    assert "hello world" in out.stdout


def test_kernel_execute_expression_repr(kernel):
    out = kernel.execute("1 + 1")
    assert out.success is True
    assert out.last_repr == "2"


def test_kernel_execute_assignment_no_repr(kernel):
    out = kernel.execute("x = 42")
    assert out.success is True
    assert out.last_repr == "" or out.last_repr is None or out.last_repr == "None"


def test_kernel_execute_exception_captured(kernel):
    out = kernel.execute("1 / 0")
    assert out.success is False
    assert "ZeroDivisionError" in out.error


def test_kernel_execute_persists_state(kernel):
    kernel.execute("foo = 99")
    out = kernel.execute("foo")
    assert out.last_repr == "99"


def test_kernel_set_context(kernel):
    kernel.set_context("target_col", "This is the brief text.")
    assert kernel.namespace["TARGET"] == "target_col"
    assert kernel.namespace["BRIEF"] == "This is the brief text."


def test_kernel_exec_count_increments(kernel):
    assert kernel._exec_count == 0
    kernel.execute("x = 1")
    assert kernel._exec_count == 1
    kernel.execute("y = 2")
    assert kernel._exec_count == 2


def test_kernel_execute_matplotlib_figure(kernel):
    out = kernel.execute(
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\n"
        "ax.plot([1, 2, 3])\n"
    )
    assert out.success is True
    assert len(out.figures) >= 1
    # Should be valid base64
    import base64
    base64.b64decode(out.figures[0])


def test_kernel_execute_syntax_error(kernel):
    out = kernel.execute("def broken(:")
    assert out.success is False
    assert out.error != ""


# ── _compile_with_repr helper ─────────────────────────────────────────────────

def test_compile_with_repr_expression():
    code = "1 + 2"
    compiled = _compile_with_repr(code)
    ns = {}
    exec(compiled, ns)  # noqa: S102
    assert ns.get("__last_repr__") == "3"


def test_compile_with_repr_statement():
    code = "x = 5"
    compiled = _compile_with_repr(code)
    ns = {}
    exec(compiled, ns)  # noqa: S102
    assert "__last_repr__" not in ns or ns.get("__last_repr__") == ""


def test_compile_with_repr_syntax_error():
    code = "def broken(:"
    # ast.parse fails → _compile_with_repr tries compile() which also raises
    with pytest.raises(SyntaxError):
        _compile_with_repr(code)
