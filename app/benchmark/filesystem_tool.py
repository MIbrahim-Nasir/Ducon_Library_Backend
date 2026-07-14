"""Safe scoped filesystem tool for dev benchmark agents."""
from __future__ import annotations

import os
import shutil
import unicodedata
from pathlib import Path, PureWindowsPath
from typing import Any


_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class FilesystemScopeError(ValueError):
    """Raised when a requested filesystem operation leaves the configured root."""


FILESYSTEM_OPERATIONS = (
    "list",
    "stat",
    "read_text",
    "write_text",
    "mkdir",
    "delete",
)


class ScopedFilesystemTool:
    """Agent-callable CRUD operations confined to one local directory."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        if not root:
            raise FilesystemScopeError("filesystem root is required")
        self.root = Path(root).expanduser().resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)
        self._root_real = self.root.resolve(strict=True)

    def _validate_subpath(self, subpath: str) -> Path:
        raw = unicodedata.normalize("NFKC", str(subpath or "")).strip()
        if not raw or raw in {".", "./"}:
            return self._root_real
        if "\x00" in raw or any(unicodedata.category(ch) == "Cc" for ch in raw):
            raise FilesystemScopeError("path contains control characters")
        win = PureWindowsPath(raw)
        if win.is_absolute() or win.drive or raw.startswith(("\\\\", "//")):
            raise FilesystemScopeError("absolute paths are not allowed")

        parts: list[str] = []
        for part in raw.replace("\\", "/").split("/"):
            if not part or part == ".":
                continue
            normalized = part.rstrip(" .")
            if normalized == ".." or part == "..":
                raise FilesystemScopeError("parent traversal is not allowed")
            if normalized.upper() in _WINDOWS_RESERVED:
                raise FilesystemScopeError(f"reserved path segment: {part}")
            parts.append(part)

        target = self._root_real.joinpath(*parts).resolve(strict=False)
        try:
            common = os.path.commonpath(
                [os.path.normcase(str(self._root_real)), os.path.normcase(str(target))]
            )
        except ValueError as exc:
            raise FilesystemScopeError("path escapes the filesystem scope") from exc
        if common != os.path.normcase(str(self._root_real)):
            raise FilesystemScopeError("path escapes the filesystem scope")
        return target

    def _relative(self, path: Path) -> str:
        return path.relative_to(self._root_real).as_posix() or "."

    def stat(self, path: str = ".") -> dict[str, Any]:
        target = self._validate_subpath(path)
        if not target.exists():
            raise FileNotFoundError(path)
        return {
            "path": self._relative(target),
            "is_file": target.is_file(),
            "is_dir": target.is_dir(),
            "size": target.stat().st_size,
        }

    def list(self, path: str = ".") -> dict[str, Any]:
        target = self._validate_subpath(path)
        if not target.exists() or not target.is_dir():
            raise NotADirectoryError(path)
        entries = []
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            resolved = child.resolve(strict=False)
            self._ensure_inside(resolved)
            entries.append(
                {
                    "name": child.name,
                    "path": self._relative(resolved),
                    "is_dir": resolved.is_dir(),
                    "size": resolved.stat().st_size if resolved.exists() else 0,
                }
            )
        return {"path": self._relative(target), "entries": entries}

    def read_text(self, path: str, *, max_chars: int = 100_000) -> dict[str, Any]:
        target = self._validate_subpath(path)
        if not target.is_file():
            raise FileNotFoundError(path)
        text = target.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_chars
        return {
            "path": self._relative(target),
            "text": text[:max_chars],
            "truncated": truncated,
        }

    def write_text(self, path: str, content: str, *, append: bool = False) -> dict[str, Any]:
        target = self._validate_subpath(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(target, mode, encoding="utf-8") as f:
            f.write(content)
        return self.stat(self._relative(target))

    def mkdir(self, path: str) -> dict[str, Any]:
        target = self._validate_subpath(path)
        target.mkdir(parents=True, exist_ok=True)
        return self.stat(self._relative(target))

    def delete(self, path: str, *, recursive: bool = False) -> dict[str, Any]:
        target = self._validate_subpath(path)
        if target == self._root_real:
            raise FilesystemScopeError("refusing to delete filesystem root")
        if not target.exists():
            return {"path": path, "deleted": False}
        if target.is_dir():
            if not recursive:
                target.rmdir()
            else:
                shutil.rmtree(target)
        else:
            target.unlink()
        return {"path": path, "deleted": True}

    def call(self, operation: str, **kwargs: Any) -> dict[str, Any]:
        op = str(operation or "").strip()
        if op == "list":
            return self.list(kwargs.get("path", "."))
        if op == "stat":
            return self.stat(kwargs.get("path", "."))
        if op == "read_text":
            return self.read_text(kwargs.get("path", ""), max_chars=int(kwargs.get("max_chars", 100_000)))
        if op == "write_text":
            return self.write_text(
                kwargs.get("path", ""),
                str(kwargs.get("content", "")),
                append=bool(kwargs.get("append", False)),
            )
        if op == "mkdir":
            return self.mkdir(kwargs.get("path", ""))
        if op == "delete":
            return self.delete(kwargs.get("path", ""), recursive=bool(kwargs.get("recursive", False)))
        raise ValueError(f"Unsupported filesystem operation: {operation}")

    def _ensure_inside(self, target: Path) -> None:
        common = os.path.commonpath(
            [os.path.normcase(str(self._root_real)), os.path.normcase(str(target))]
        )
        if common != os.path.normcase(str(self._root_real)):
            raise FilesystemScopeError("path escapes the filesystem scope")
