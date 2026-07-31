# Documentation source

This directory is the complete source and build boundary for the
SuperNPUBench website. Published pages and assets live in `content/`; the
MkDocs configuration, pinned Python dependencies, generated catalogs,
generators, and verification tools live beside it.

The repository root is needed only when a generated benchmark page reads a
benchmark manifest or source file. No website configuration or generator is
stored outside `docs/`.

## Local build

From the repository root:

```console
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r docs/requirements.txt
docs/build.sh
```

The strict site is written to `site/`. Run the local server with:

```console
docs/build.sh serve
```

MkDocs then serves the manual at `http://127.0.0.1:8000/SuperNPUBench/`.

## Generated references

Regenerate source-backed navigation after benchmark or intrinsic changes:

```console
python3 docs/scripts/sync_golden_manual.py --write
python3 docs/scripts/generate_benchmark_manual.py
python3 docs/scripts/generate_deepseek_manifest.py
python3 docs/scripts/generate_deepseek_manual.py
```

`docs/build.sh` verifies that checked-in generated pages agree with their
catalogs and that all links in the built site resolve.

## Cross-stack parity

When a checkout is nested at `workloads/SuperNPUBench` in the Linx
superproject, validate the manual's operation inventory against the canonical
v0.57 state before publication:

```console
export LINX_ROOT=/path/to/linx-isa
python3 "$LINX_ROOT/tools/isa/build_golden.py" --profile v0.57 --check
python3 "$LINX_ROOT/tools/isa/validate_spec.py" --profile v0.57
python3 "$LINX_ROOT/tools/isa/check_canonical_v057.py" --root "$LINX_ROOT"
python3 "$LINX_ROOT/tools/isa/check_pto_v057_manifest.py" --root "$LINX_ROOT"
python3 docs/scripts/sync_golden_manual.py --check --linx-root "$LINX_ROOT"
```

Use the compiler and models pinned by that same superproject. Rebuild the
in-repository Clang before compiling representative cases, and run QEMU before
promoting an ELF to plain `gfsim -f <elf>`.
