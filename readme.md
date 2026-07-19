# CPGen

CPGen is an indentation-based Domain-Specific Language (DSL) that compiles directly into Python abstract syntax trees (AST) to generate structured test cases for competitive programming problems.

## Features

* **Indentation-based layout:** Employs a custom lexer that manages an indentation stack to emit clean `INDENT` and `DEDENT` structural tokens.
* **Hygiene-aware compilation:** Automatically namespaces user-declared variables with a `_u_` prefix to prevent collision with Python builtins or the `random` library.
* **Deterministic scoping:** Performs semantic variable validation inline during the parsing pass.
* **Native VS Code integration:** Bundled with a custom TextMate grammar configuration for full syntax highlighting.

## Language Syntax

CPGen allows declarations of randomized integers, floats, strings, and multi-dimensional arrays, using previously declared variables as constraints.

```text
# Primitives and Range Bounds
nodes:   int from [2..500]
edges:   int from [1..1000]

# Arithmetic Expressions inside bounds
low:     int from [1..nodes*2]

# Character Sets and Strings
dna:     string(20) from ['A'..'A'] + "CGT"
word:    string(nodes) from ['a'..'z']

# Nested Matrix / Array Declarations
matrix:  array(nodes) of array(edges) of int from [-100..100]

# Execution Loops
@println(nodes, edges)
repeat edges:
    u: int from [1..nodes]
    v: int from [1..nodes]
    w: int from [1..100000]
    @println(u, v, w)
```

## Pipeline Architecture

The compiler framework processes source code across three main boundaries:
1. **`lexer.py`**: Converts raw characters into a flat token stream while evaluating line-by-line whitespace levels.
2. **`parser.py`**: Executes a top-down recursive descent parse, running a Pratt/precedence-climbing algorithm to fold expression logic into concrete Python AST nodes.
3. **`main.py`**: The CLI orchestrator that calls `ast.unparse()` to output structural, runnable Python code.

## Usage

Run the compiler via the command line interface provided in `Compiler/main.py`:

**Print compiled Python code directly to stdout:**
```bash
python Compiler/main.py script.cpgen
```

**Save compiled code directly to a file:**
```bash
python Compiler/main.py script.cpgen -o generator.py
```

**Compile and immediately execute the test case generator:**
```bash
python Compiler/main.py script.cpgen -e
```

## Running Tests

The compiler includes an end-to-end black-box integration test suite located in `Compiler/tests/tests_unit.py`. The suite verifies runtime behavioral output and side-effects rather than strict AST mapping.

To run the verification checks:
```bash
python -m unittest Compiler/tests/tests_unit.py
```
