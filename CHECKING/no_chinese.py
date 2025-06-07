import os
import re
import sys


def has_chinese(text):
    return re.search(r"[\u4e00-\u9fff]", text)


def check_chinese_in_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if has_chinese(line):
                print(f"{filepath}:{lineno}: Chineses Letter -> {line.strip()}")
                return True
    return False


def main():
    failed = False
    for root, _, files in os.walk("./src"):
        if ".venv" in root or "site-packages" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                if check_chinese_in_file(path):
                    failed = True
    if failed:
        print("checking no_chinese.py failed: Chinese characters found in code.")
        sys.exit(1)
    else:
        print("checking no_chinese.py pass: No Chinese characters found in code.")


if __name__ == "__main__":
    main()
