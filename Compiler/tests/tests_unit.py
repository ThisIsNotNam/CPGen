"""
Unit tests for the CPGen pipeline, exercised end-to-end through the public
build() + exec() boundary.

Philosophy: these tests deliberately avoid inspecting the generated AST's
internal shape (e.g. "is this node a Constant or a Call", "is this folded
at parse time or left as a BinOp"). Whether a value gets produced via
constant-folding, a runtime int()/float() wrap, or random.randint vs.
random.uniform is an implementation detail of *how* the compiler gets
there — it shouldn't make a test pass or fail. Only the *observable
behavior* (the value(s) that end up in the namespace after running the
generated code, and what gets printed to stdout) is the actual contract,
and that's what's asserted on below.

The only AST-level check kept is test_output_is_valid_python, which checks
the *public* contract that build()'s output is valid Python — not CPGen's
internal parser structures.
"""

import ast
import contextlib
import io
import sys
import os
import unittest
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cpgen

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def run(src: str) -> Tuple[Dict[str, Any], str]:
    """Build *src*, execute it once, and return (namespace, captured stdout)."""
    code = cpgen.build(src)
    ns: Dict[str, Any] = {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, ns)  # noqa: S102
    return ns, buf.getvalue()


def sample(src: str, var: str, trials: int = 50) -> List[Any]:
    """
    Build *src* once, then re-execute it *trials* times, collecting the
    runtime value of *var* (e.g. "_u_x") on each run.

    Used for randomized constructs (rand ranges, charset gen, arrays) so
    that a single lucky/unlucky draw can't hide an off-by-one bound error.
    """
    code = cpgen.build(src)
    values: List[Any] = []
    for _ in range(trials):
        ns: Dict[str, Any] = {}
        exec(code, ns)  # noqa: S102
        values.append(ns[var])
    return values


# ─────────────────────────────────────────────────────────────────────────────
# Int literals
# ─────────────────────────────────────────────────────────────────────────────


class TestIntLiteral(unittest.TestCase):
    def test_plain(self):
        ns, _ = run("x: int = 42\n")
        self.assertEqual(ns["_u_x"], 42)
        self.assertIsInstance(ns["_u_x"], int)

    def test_zero(self):
        ns, _ = run("x: int = 0\n")
        self.assertEqual(ns["_u_x"], 0)

    def test_negative(self):
        ns, _ = run("x: int = -7\n")
        self.assertEqual(ns["_u_x"], -7)

    def test_scientific_notation(self):
        """int = 2e3 → runtime value 2000, as a real int."""
        ns, _ = run("x: int = 2e3\n")
        self.assertEqual(ns["_u_x"], 2000)
        self.assertIsInstance(ns["_u_x"], int)

    def test_float_literal_truncates(self):
        ns, _ = run("x: int = 3.14\n")
        self.assertEqual(ns["_u_x"], 3)  # truncates, not rounds
        self.assertIsInstance(ns["_u_x"], int)

    def test_float_variable_truncates(self):
        """Assigning a float-typed *variable* into an int-typed one must
        also truncate at runtime, not just float literals."""
        ns, _ = run("t: float = 1.123\nn: int = t\n")
        self.assertEqual(ns["_u_n"], 1)
        self.assertIsInstance(ns["_u_n"], int)


# ─────────────────────────────────────────────────────────────────────────────
# Float literals
# ─────────────────────────────────────────────────────────────────────────────


class TestFloatLiteral(unittest.TestCase):
    def test_plain(self):
        ns, _ = run("x: float = 3.14\n")
        self.assertAlmostEqual(ns["_u_x"], 3.14)
        self.assertIsInstance(ns["_u_x"], float)

    def test_scientific(self):
        ns, _ = run("x: float = 1e2\n")
        self.assertAlmostEqual(ns["_u_x"], 100.0)
        self.assertIsInstance(ns["_u_x"], float)

    def test_zero(self):
        ns, _ = run("x: float = 0.0\n")
        self.assertAlmostEqual(ns["_u_x"], 0.0)
        self.assertIsInstance(ns["_u_x"], float)

    def test_int_literal_becomes_float(self):
        """A plain int-looking literal assigned to a float-typed variable
        must still come out as a real float at runtime."""
        ns, _ = run("x: float = 42\n")
        self.assertEqual(ns["_u_x"], 42.0)
        self.assertIsInstance(ns["_u_x"], float)

    def test_int_variable_becomes_float(self):
        ns, _ = run("n: int = 5\nx: float = n\n")
        self.assertEqual(ns["_u_x"], 5.0)
        self.assertIsInstance(ns["_u_x"], float)


# ─────────────────────────────────────────────────────────────────────────────
# String literals
# ─────────────────────────────────────────────────────────────────────────────


class TestStringLiteral(unittest.TestCase):
    def test_plain(self):
        ns, _ = run('x: string = "hello"\n')
        self.assertEqual(ns["_u_x"], "hello")

    def test_empty(self):
        ns, _ = run('x: string = ""\n')
        self.assertEqual(ns["_u_x"], "")


# ─────────────────────────────────────────────────────────────────────────────
# Int rand range  →  random.randint(lo, hi)
# ─────────────────────────────────────────────────────────────────────────────


class TestIntRandRange(unittest.TestCase):
    def test_within_bounds_and_type(self):
        for v in sample("x: int from [1..10]\n", "_u_x"):
            self.assertIsInstance(v, int)
            self.assertGreaterEqual(v, 1)
            self.assertLessEqual(v, 10)

    def test_variable_bound(self):
        for v in sample("n: int = 5\nx: int from [1..n]\n", "_u_x"):
            self.assertIsInstance(v, int)
            self.assertGreaterEqual(v, 1)
            self.assertLessEqual(v, 5)

    def test_min_exceeds_max_raises(self):
        with self.assertRaises(ValueError):
            cpgen.build("x: int from [10..1]\n")


# ─────────────────────────────────────────────────────────────────────────────
# Float rand range  →  random.uniform(lo, hi)
# ─────────────────────────────────────────────────────────────────────────────


class TestFloatRandRange(unittest.TestCase):
    def test_within_bounds_and_type(self):
        for v in sample("x: float from [1.0..9.9]\n", "_u_x"):
            self.assertIsInstance(v, float)
            self.assertGreaterEqual(v, 1.0)
            self.assertLessEqual(v, 9.9)

    def test_zero_to_one_bounds(self):
        for v in sample("x: float from [0.0..1.0]\n", "_u_x"):
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_min_exceeds_max_raises(self):
        with self.assertRaises(ValueError):
            cpgen.build("x: float from [9.9..1.0]\n")


# ─────────────────────────────────────────────────────────────────────────────
# String rand gen  →  ''.join(random.choices(..., k=n))
# ─────────────────────────────────────────────────────────────────────────────


class TestStringRandGen(unittest.TestCase):
    def test_charset_range(self):
        for v in sample("x: string(5) from ['a'..'z']\n", "_u_x"):
            self.assertEqual(len(v), 5)
            self.assertTrue(all("a" <= c <= "z" for c in v))

    def test_length_matches_request(self):
        for v in sample("x: string(10) from ['a'..'z']\n", "_u_x"):
            self.assertEqual(len(v), 10)

    def test_literal_charset(self):
        for v in sample("x: string(3) from 'abc'\n", "_u_x"):
            self.assertEqual(len(v), 3)
            self.assertTrue(all(c in "abc" for c in v))

    def test_negative_length_raises(self):
        with self.assertRaises(ValueError):
            cpgen.build("x: string(-1) from ['a'..'z']\n")


# ─────────────────────────────────────────────────────────────────────────────
# Array  →  list comprehension
# ─────────────────────────────────────────────────────────────────────────────


class TestArrayRandGen(unittest.TestCase):
    def test_length_and_elements(self):
        for v in sample("x: array(5) of int from [1..10]\n", "_u_x"):
            self.assertEqual(len(v), 5)
            for elem in v:
                self.assertIsInstance(elem, int)
                self.assertGreaterEqual(elem, 1)
                self.assertLessEqual(elem, 10)

    def test_size_zero_raises(self):
        with self.assertRaises(ValueError):
            cpgen.build("x: array(0) of int from [1..10]\n")

    def test_variable_size(self):
        ns, _ = run("n: int = 5\nx: array(n) of int from [1..10]\n")
        self.assertEqual(len(ns["_u_x"]), 5)


# ─────────────────────────────────────────────────────────────────────────────
# Repeat  →  for _ in range(n):
# ─────────────────────────────────────────────────────────────────────────────


class TestRepeat(unittest.TestCase):
    def test_body_runs_n_times(self):
        _, out = run("repeat 3:\n    x: int = 1\n    @println(x)\n")
        self.assertEqual(out, "1\n1\n1\n")

    def test_repeat_zero_runs_nothing(self):
        _, out = run("repeat 0:\n    x: int = 1\n    @println(x)\n")
        self.assertEqual(out, "")

    def test_repeat_negative_raises(self):
        with self.assertRaises(ValueError):
            cpgen.build("repeat -1:\n    x: int from [1..10]\n")

    def test_body_with_multiple_statements_runs_each_iteration(self):
        src = "repeat 2:\n    x: int = 1\n    y: int = 3\n    @println(y)\n"
        _, out = run(src)
        self.assertEqual(out, "3\n3\n")


# ─────────────────────────────────────────────────────────────────────────────
# Print / println
# ─────────────────────────────────────────────────────────────────────────────


class TestPrint(unittest.TestCase):
    def test_print_no_trailing_newline(self):
        _, out = run("x: int = 1\n@print(x)\n")
        self.assertEqual(out, "1")

    def test_println_has_trailing_newline(self):
        _, out = run("x: int = 1\n@println(x)\n")
        self.assertEqual(out, "1\n")

    def test_print_undefined_raises(self):
        with self.assertRaises(NameError):
            cpgen.build("@print(y)\n")

    def test_print_array_unpacks_elements(self):
        _, out = run("x: array(3) of int from [1..9]\n@println(x)\n")
        parts = out.strip().split()
        self.assertEqual(len(parts), 3)
        for p in parts:
            self.assertTrue(1 <= int(p) <= 9)


# ─────────────────────────────────────────────────────────────────────────────
# Undefined / redeclared variables
# ─────────────────────────────────────────────────────────────────────────────


class TestVariableScoping(unittest.TestCase):
    def test_undefined_in_bound_raises(self):
        with self.assertRaises(NameError):
            cpgen.build("x: int from [1..n]\n")

    def test_redeclare_raises(self):
        with self.assertRaises(NameError):
            cpgen.build("x: int = 1\nx: int = 2\n")

    def test_forward_reference_raises(self):
        with self.assertRaises(NameError):
            cpgen.build("x: int from [1..y]\ny: int = 5\n")

    def test_builtin_name_allowed(self):
        """'list' is a Python builtin but should still work as a CPGen
        variable name."""
        _, out = run("list: int = 1\n@println(list)\n")
        self.assertEqual(out, "1\n")

    def test_builtin_range_allowed(self):
        """'range' is a Python builtin but should still work as a CPGen
        variable name."""
        ns, out = run("range: int from [1..10]\n@println(range)\n")
        self.assertEqual(out.strip(), str(ns["_u_range"]))

    def test_valid_reference(self):
        """Variable declared before use in a bound is accepted, and the
        bound it contributes to is actually respected."""
        for v in sample("n: int = 10\nx: int from [1..n]\n", "_u_x"):
            self.assertGreaterEqual(v, 1)
            self.assertLessEqual(v, 10)


# ─────────────────────────────────────────────────────────────────────────────
# Expression parser
# ─────────────────────────────────────────────────────────────────────────────


class TestExprParser(unittest.TestCase):
    def test_constant_expr_bound(self):
        """2 * 3 as an upper bound should behave as 6 — whether or not
        it's folded at parse time is irrelevant."""
        for v in sample("x: int from [1..2*3]\n", "_u_x"):
            self.assertGreaterEqual(v, 1)
            self.assertLessEqual(v, 6)

    def test_arithmetic_with_variable_bound(self):
        for v in sample("n: int = 5\nx: int from [1..n*2]\n", "_u_x"):
            self.assertGreaterEqual(v, 1)
            self.assertLessEqual(v, 10)

    def test_parenthesised_expr_bound(self):
        """(1 + 2) * 3 should behave as 9."""
        for v in sample("x: int from [0..(1+2)*3]\n", "_u_x"):
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, 9)

    def test_invalid_token_in_expr_raises(self):
        with self.assertRaises(SyntaxError):
            cpgen.build('x: int from [1.."bad"]\n')

    def test_output_is_valid_python(self):
        """Public contract: build() always returns syntactically valid
        Python, regardless of how CPGen internally represents it."""
        code = cpgen.build("x: int from [1..10]\n@println(x)\n")
        self.assertIsInstance(ast.parse(code), ast.Module)

    def test_output_is_executable(self):
        _, out = run("x: int = 42\n@println(x)\n")
        self.assertEqual(out, "42\n")

    def test_import_random_present_when_needed(self):
        self.assertIn("import random", cpgen.build("x: int from [1..5]\n"))


# ─────────────────────────────────────────────────────────────────────────────
# Reassignment  (no type annotation: x = <expr>)
# ─────────────────────────────────────────────────────────────────────────────


class TestReassignment(unittest.TestCase):
    def test_basic_reassignment(self):
        ns, _ = run("x: int = 1\nx = 5\n")
        self.assertEqual(ns["_u_x"], 5)

    def test_reassign_with_expression(self):
        ns, _ = run("x: int = 1\ny: int = 2\nx = y + 3\n")
        self.assertEqual(ns["_u_x"], 5)

    def test_reassign_truncates_float_into_int(self):
        ns, _ = run("x: int = 1\nx = 3.14\n")
        self.assertEqual(ns["_u_x"], 3)
        self.assertIsInstance(ns["_u_x"], int)

    def test_reassign_promotes_int_into_float(self):
        ns, _ = run("x: float = 1.0\ny: int = 5\nx = y\n")
        self.assertEqual(ns["_u_x"], 5.0)
        self.assertIsInstance(ns["_u_x"], float)

    def test_reassign_string(self):
        ns, _ = run('x: string = "a"\nx = "b"\n')
        self.assertEqual(ns["_u_x"], "b")

    def test_reassign_undeclared_raises(self):
        with self.assertRaises(NameError):
            cpgen.build("x = 5\n")

    def test_reassign_array_raises(self):
        with self.assertRaises(NotImplementedError):
            cpgen.build("x: array(2) of int from [1..5]\nx = x\n")

    def test_reassign_int_with_string_raises(self):
        with self.assertRaises(TypeError):
            cpgen.build('x: int = 1\ny: string = "a"\nx = y\n')

    def test_reassign_string_with_number_raises(self):
        with self.assertRaises(TypeError):
            cpgen.build('x: string = "a"\nx = 5\n')

    def test_reassign_does_not_clear_declaration_state(self):
        """Reassigning shouldn't make the name re-declarable."""
        with self.assertRaises(NameError):
            cpgen.build("x: int = 1\nx = 2\nx: int = 3\n")

    def test_reassign_accumulates_in_repeat(self):
        _, out = run("x: int = 0\nrepeat 3:\n    x = x + 1\n@println(x)\n")
        self.assertEqual(out, "3\n")


if __name__ == "__main__":
    unittest.main()
