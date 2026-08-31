# -*- coding: utf-8 -*-
"""命令行入口：python cli.py 输入文件夹 [输出文件夹] [ru es pt]"""
import os
import sys

from core.batch import run_batch


def main():
    if len(sys.argv) < 2:
        print("用法: python cli.py 输入文件夹 [输出文件夹] [ru es pt]")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2] in ("ru", "es", "pt", "en") \
        else os.path.join(input_dir, "翻译输出")
    targets = [a for a in sys.argv[2:] if a in ("ru", "es", "pt", "en")] or ["ru", "es", "pt"]
    result = run_batch(input_dir, output_dir, targets)
    print(f"完成: {result}")


if __name__ == "__main__":
    main()
