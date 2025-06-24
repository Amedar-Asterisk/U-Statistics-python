import os
import ast
import sys


def has_print(node):
    """
    Check if the AST node contains a print statement.
    """
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            return True
    for child in ast.iter_child_nodes(node):
        if has_print(child):
            return True
    return False


def check_print_in_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
            if has_print(tree):
                print(f"{filepath}: Contains print statement")
                return True
        except SyntaxError:
            print(f"Syntax error in file: {filepath}")
            return False
    return False


def main():
    failed = False
    for root, _, files in os.walk("./src"):
        if ".venv" in root or "site-packages" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                if check_print_in_file(path):
                    failed = True

    if failed:
        print("checking no_print.py failed: print statements found in code.")
        sys.exit(1)
    else:
        print("checking no_print.py pass: No print statements found in code.")


if __name__ == "__main__":
    main()
