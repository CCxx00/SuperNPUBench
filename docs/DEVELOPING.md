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
