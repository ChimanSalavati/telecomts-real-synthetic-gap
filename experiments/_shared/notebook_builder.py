"""Tiny helper for building Jupyter notebook (.ipynb) files programmatically.

Usage::

    from _shared.notebook_builder import NotebookBuilder
    nb = NotebookBuilder("E1: Dataset and split audit")
    nb.md("# Header\nIntro text...")
    nb.code("import numpy as np\nprint('hi')")
    nb.write("E1_dataset_split_leakage_audit.ipynb")

The output uses nbformat v4 with empty outputs so the file can be opened
without modification in Jupyter or VS Code.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Iterable


def _new_cell_id() -> str:
    return uuid.uuid4().hex[:12]


def _split_lines(text: str) -> list[str]:
    """nbformat stores ``source`` as a list of lines, each ending with ``\n``
    except possibly the last one.
    """
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    return lines


class NotebookBuilder:
    def __init__(self, title: str | None = None) -> None:
        self._cells: list[dict] = []
        self._title = title

    def md(self, source: str) -> "NotebookBuilder":
        self._cells.append({
            "cell_type": "markdown",
            "id": _new_cell_id(),
            "metadata": {},
            "source": _split_lines(source),
        })
        return self

    def code(self, source: str) -> "NotebookBuilder":
        self._cells.append({
            "cell_type": "code",
            "id": _new_cell_id(),
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _split_lines(source),
        })
        return self

    def extend_md(self, sources: Iterable[str]) -> "NotebookBuilder":
        for s in sources:
            self.md(s)
        return self

    def extend_code(self, sources: Iterable[str]) -> "NotebookBuilder":
        for s in sources:
            self.code(s)
        return self

    def to_dict(self) -> dict:
        return {
            "cells": self._cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10",
                },
                "title": self._title or "",
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(self.to_dict(), f, indent=1)
        return p


__all__ = ["NotebookBuilder"]
