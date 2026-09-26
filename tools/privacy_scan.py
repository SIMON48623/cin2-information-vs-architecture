from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess


MAX_SIZE = 1024 * 1024
FORBIDDEN_EXTENSIONS = {".xlsx", ".xls", ".npz", ".pt", ".ckpt"}


def tracked_or_all(root: Path):
    try:
        output = subprocess.check_output(["git", "ls-files"], cwd=root, text=True)
        paths = [root / line for line in output.splitlines() if line]
        if paths: return paths
    except Exception:
        pass
    return [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]


def scan(root: Path):
    findings = []
    absolute_markers = ["C:" + chr(92) + chr(92) + "Users", "Documents" + chr(92), "/" + "home/", "/" + "mnt/"]
    email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    phone = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
    qq = re.compile(r"(?i)qq\s*[:=]?\s*[1-9]\d{4,11}")
    chinese_name = re.compile(r"(?:姓名|患者)\s*[:：]\s*[\u4e00-\u9fff]{2,4}")
    allowed_unicode = {"docs/prespecification/A_analysis_plan_prespecified.md"}
    for path in tracked_or_all(root):
        relative = path.relative_to(root).as_posix()
        if path.stat().st_size > MAX_SIZE: findings.append(f"oversize: {relative}")
        if path.suffix.lower() in FORBIDDEN_EXTENSIONS: findings.append(f"forbidden extension: {relative}")
        try: text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError: continue
        for marker in absolute_markers:
            if marker in text: findings.append(f"absolute path marker in {relative}")
        if email.search(text): findings.append(f"email in {relative}")
        if phone.search(text): findings.append(f"phone-like number in {relative}")
        if qq.search(text): findings.append(f"QQ-like identifier in {relative}")
        if relative not in allowed_unicode and chinese_name.search(text): findings.append(f"Chinese name-like value in {relative}")
        if re.search(r"(?i)(patient_name|patient_id)\s*[,=:]\s*[^\s,}\]]+", text):
            findings.append(f"possible direct identifier value in {relative}")
    return sorted(set(findings))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(); findings = scan(args.root.resolve())
    if findings:
        print("PRIVACY SCAN FAILED")
        for finding in findings: print("-", finding)
        raise SystemExit(1)
    print(f"PRIVACY SCAN PASSED: {len(tracked_or_all(args.root.resolve()))} files checked")


if __name__ == "__main__": main()

