import glob
import json
import os
import sys

FOLDER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(FOLDER))

from kan_chiong_driver import solve


def main():
    args = sys.argv[1:]
    if args:
        test_files = [
            path if os.path.isabs(path) else os.path.join(FOLDER, path)
            for path in args
        ]
    else:
        test_files = sorted(glob.glob(os.path.join(FOLDER, "test_case_*.json")))

    all_pass = True
    for path in test_files:
        with open(path, "r", encoding="utf-8") as f:
            case = json.load(f)
        actual = json.loads(solve(json.dumps(case["input"])))
        ok = actual == case["expected_output"]
        all_pass &= ok
        name = os.path.basename(path)
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
        print("  expected:", json.dumps(case["expected_output"]))
        print("  actual:  ", json.dumps(actual))

    print("\nALL PASS" if all_pass else "\nSOME FAILED")


if __name__ == "__main__":
    main()
