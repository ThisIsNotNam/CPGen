import sys
import ast
import argparse
import builtins
import cpgen


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cpgen",
        description="Compile a CPGen DSL file into a Python test-case generator.",
    )
    p.add_argument("filepath", help="Path to the .cpgen source file")
    p.add_argument("-o", "--output", type=str, help="Output file path")
    p.add_argument("-e", "--eval", action="store_true")  # Defaults to False
    return p


def main():
    args = _build_arg_parser().parse_args()

    with open(args.filepath) as f:
        source = f.read()

    try:
        module = cpgen._parse(source)
        if args.eval:
            exec(builtins.compile(module, "<cpgen>", "exec"), {})  # noqa: S102
        elif args.output:
            with open(args.output, "w") as f:
                f.write(ast.unparse(module))
        else:
            print(ast.unparse(module))
    except (SyntaxError, NameError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
