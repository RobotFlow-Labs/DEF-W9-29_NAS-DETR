from __future__ import annotations

from pathlib import Path


def main() -> None:
    required = [
        Path("ASSETS.md"),
        Path("papers/2505.06694.pdf"),
        Path("prds/README.md"),
        Path("tasks/INDEX.md"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required assets: {missing}")
    print("ASSET_CHECK_OK")


if __name__ == "__main__":
    main()
