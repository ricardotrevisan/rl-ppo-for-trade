#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path | None:
    p = start.resolve()
    for candidate in [p] + list(p.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def collect_files(data_dir: Path, exts: list[str]) -> list[Path]:
    files = set()
    for ext in exts:
        ext = ext if ext.startswith(".") else f".{ext}"
        files.update(p for p in data_dir.rglob(f"*{ext}") if p.is_file())
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete selected file types (default: .csv, .png) inside the data directory, recursively."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Target data directory (default: data)"
    )
    parser.add_argument(
        "-e", "--extensions",
        nargs="*",
        default=[".csv", ".png"],
        help="File extensions to delete (e.g., -e .csv .png .jpg)"
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Only list files, do not delete"
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Do not prompt for confirmation"
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return 0
    if not data_dir.is_dir():
        print(f"❌ Not a directory: {data_dir}")
        return 1

    # Safety: ensure operation is inside repo root
    repo_root = find_repo_root(Path.cwd())
    if repo_root and repo_root not in [data_dir] + list(data_dir.parents):
        print("⚠️ Refusing to operate outside the repository root.")
        print(f"Repo root: {repo_root}")
        print(f"Target dir: {data_dir}")
        return 1

    files = collect_files(data_dir, args.extensions)

    if not files:
        print("✅ No matching files found.")
        return 0

    print(f"📁 Target directory: {data_dir}")
    print(f"📄 Extensions: {', '.join(args.extensions)}")
    print(f"🔍 Found {len(files)} files.")

    if args.dry_run:
        for p in files:
            print(p)
        print("💡 Dry run complete. No files deleted.")
        return 0

    if not args.yes:
        preview = files[:10]
        print("🧾 Preview (first up to 10 files):")
        for p in preview:
            print(f"  {p}")
        if len(files) > len(preview):
            print(f"  ... and {len(files) - len(preview)} more")

        ans = input("Type 'yes' to confirm deletion: ").strip().lower()
        if ans != "yes":
            print("🚫 Aborted by user.")
            return 1

    deleted, errors = 0, 0
    for p in files:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            errors += 1
            print(f"❗ Failed to delete {p}: {e}")

    print(f"✅ Deleted: {deleted} files. ❌ Errors: {errors}.")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
