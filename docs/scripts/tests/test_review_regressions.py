from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generator = load_module(
    REPO_ROOT / "docs/scripts/generate_benchmark_manual.py",
    "generate_benchmark_manual_under_test",
)
verifier = load_module(
    REPO_ROOT / "docs/scripts/verify_golden_manual.py",
    "verify_golden_manual_under_test",
)


class ReviewRegressionTests(unittest.TestCase):
    def init_git_repo(self, root: Path) -> None:
        subprocess.run(["git", "init", "-q", str(root)], check=True)

    def test_provenance_ignores_untracked_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.init_git_repo(root)
            (root / "docs/content").mkdir(parents=True)
            (root / "README.md").write_text("# Reader documentation\n", encoding="utf-8")
            (root / "docs/content/guide.md").write_text("# Guide\n", encoding="utf-8")
            (root / "local-notes.md").write_text("a" * 40 + "\n", encoding="utf-8")
            subprocess.run(
                ["git", "-C", str(root), "add", "README.md", "docs/content/guide.md"],
                check=True,
            )

            errors: list[str] = []
            verifier.check_markdown_provenance(root, errors)

            self.assertEqual([], errors)

    def test_provenance_checks_tracked_reader_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.init_git_repo(root)
            (root / "docs/content").mkdir(parents=True)
            bad_page = root / "docs/content/guide.md"
            bad_page.write_text("b" * 40 + "\n", encoding="utf-8")
            subprocess.run(
                ["git", "-C", str(root), "add", "docs/content/guide.md"],
                check=True,
            )

            errors: list[str] = []
            verifier.check_markdown_provenance(root, errors)

            self.assertEqual(
                ["docs/content/guide.md: exposes a commit ID"],
                errors,
            )

    def test_data_inventory_contains_only_tracked_inputs(self) -> None:
        self.assertTrue(
            hasattr(generator, "tracked_data_files"),
            "generator must expose tracked_data_files",
        )
        if not hasattr(generator, "tracked_data_files"):
            return

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.init_git_repo(root)
            case_dir = root / "benchmark/case"
            data_dir = case_dir / "implementation/data_obj"
            data_dir.mkdir(parents=True)
            tracked = data_dir / "build_data_obj.sh"
            ignored = data_dir / "generated.data"
            tracked.write_text("#!/bin/sh\n", encoding="utf-8")
            ignored.write_text("generated\n", encoding="utf-8")
            (root / ".gitignore").write_text("*.data\n", encoding="utf-8")
            subprocess.run(
                ["git", "-C", str(root), "add", ".gitignore", str(tracked)],
                check=True,
            )

            self.assertEqual(
                [tracked],
                generator.tracked_data_files(root, case_dir),
            )

    def test_generator_writes_redirect_for_legacy_hashed_route(self) -> None:
        self.assertTrue(
            hasattr(generator, "write_legacy_redirect"),
            "generator must expose write_legacy_redirect",
        )
        if not hasattr(generator, "write_legacy_redirect"):
            return

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "benchmark/one-level-arch/test/kernel/matmul/src/matmul.cpp"
            source.parent.mkdir(parents=True)
            source.write_text("// source\n", encoding="utf-8")
            output = root / "docs/content/benchmarks/catalog/one-level/matmul"
            output.mkdir(parents=True)

            alias = generator.write_legacy_redirect(
                output,
                root,
                source,
                "matmul.md",
                "Tiled matrix multiplication",
            )

            self.assertEqual(
                output / "matmul-a008ac76/index.html",
                alias,
            )
            text = alias.read_text(encoding="utf-8")
            self.assertIn('http-equiv="refresh" content="0; url=../matmul/"', text)
            self.assertIn('rel="canonical" href="../matmul/"', text)

    def test_four_operand_matmul_acc_seeds_destination_from_previous(self) -> None:
        compiler = shutil.which("c++")
        self.assertIsNotNone(compiler, "a C++ compiler is required for adapter tests")
        if compiler is None:
            return

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            include = root / "include/common"
            include.mkdir(parents=True)
            (include / "pto_tileop.hpp").write_text(
                """
#pragma once
enum class BLayout { RowMajor };
enum class Location { Vec };
template <Location, typename T, int Rows, int Cols, BLayout, int, int>
struct Tile {};
struct global_iterator {};
template <typename... Args> void TADD(Args &...) {}
template <typename... Args> void TADDS(Args &...) {}
template <typename... Args> void TCVT(Args &...) {}
template <typename... Args> void TLOAD(Args &...) {}
template <typename... Args> void TMAXS(Args &...) {}
template <typename... Args> void TMULS(Args &...) {}
template <typename... Args> void TROWSUM(Args &...) {}
template <typename... Args> void TSTORE(Args &...) {}
template <typename Dst, typename Src>
void TCOPY(Dst &dst, Src &src) { dst.value = src.value; }
template <typename Dst, typename Lhs, typename Rhs>
void MATMUL(Dst &dst, Lhs &lhs, Rhs &rhs) {
  dst.value = lhs.value * rhs.value;
}
template <typename Dst, typename Lhs, typename Rhs>
void MATMACC(Dst &dst, Lhs &lhs, Rhs &rhs) {
  dst.value += lhs.value * rhs.value;
}
""",
                encoding="utf-8",
            )
            source = root / "test.cpp"
            source.write_text(
                """
#include <pto_kernel/tile.hpp>
struct ValueTile { int value; };
int main() {
  ValueTile dst{100};
  ValueTile previous{7};
  ValueTile lhs{3};
  ValueTile rhs{4};
  pto::TMATMUL_ACC(dst, previous, lhs, rhs);
  return dst.value == 19 ? 0 : 1;
}
""",
                encoding="utf-8",
            )
            binary = root / "adapter-test"
            subprocess.run(
                [
                    compiler,
                    "-std=c++20",
                    "-I",
                    str(root / "include"),
                    "-I",
                    str(REPO_ROOT / "docs/content/examples/include"),
                    str(source),
                    "-o",
                    str(binary),
                ],
                check=True,
            )

            result = subprocess.run([str(binary)], check=False)

            self.assertEqual(0, result.returncode)


if __name__ == "__main__":
    unittest.main()
