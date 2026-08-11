#!/usr/bin/env python3
"""Run a matmul ELF with gfrun and compare its output with PyTorch.

The ELF must be built with ``res_check=on`` so that the kernel reads
``src0.bin``/``src1.bin`` and writes ``res.bin`` under its CHK_DIR.

Example:

  python3 golden_cmp.py --ones -d /absolute/path/to/matmul_MASK_...elf
"""

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:
    raise SystemExit("golden_cmp.py requires PyTorch: pip install torch") from exc


DEFAULT_GFRUN = "/Users/blacktraker/Programming/gitproj/DV4/SuperScalarModel/bin/gfrun"
DEFAULT_GFRUN_ARGS = "-t 1 -f"
SCRIPT_DIR = Path(__file__).resolve().parent
ONE_LEVEL_ROOT = SCRIPT_DIR.parents[3]
COMPARE_ROOT = ONE_LEVEL_ROOT / "compare"


def extract_shape(elf: Path):
    """Extract mode and M/N/K/tM/tN/tK from a matmul ELF name."""
    name = elf.stem
    match = re.search(
        r"_M(?P<M>\d+)_N(?P<N>\d+)_K(?P<K>\d+)"
        r"_tM(?P<tM>\d+)_tN(?P<tN>\d+)_tK(?P<tK>\d+)$",
        name,
    )
    if not match:
        raise ValueError(f"cannot parse matmul shape from ELF name: {name}")

    if "MASK_FP32" in name or "MATMUL_VEC" in name:
        dtype = "fp32"
    elif "MASK_FP16" in name:
        dtype = "fp16"
    elif "MASK_FP8" in name:
        dtype = "fp8_e4m3"
    else:
        raise ValueError(
            "unsupported matmul mode; expected MASK_FP32, MASK_FP16, "
            "MASK_FP8, or MATMUL_VEC"
        )

    shape = {key: int(value) for key, value in match.groupdict().items()}
    shape.update({"name": name, "dtype": dtype})
    return shape


def quantize_inputs(a: torch.Tensor, b: torch.Tensor, dtype: str):
    """Quantize inputs exactly as stored in the kernel input files."""
    if dtype == "fp32":
        return a.float(), b.float(), a.numpy(), b.numpy()
    if dtype == "fp16":
        aq = a.half()
        bq = b.half()
        return aq.float(), bq.float(), aq.numpy(), bq.numpy()
    if dtype == "fp8_e4m3":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("this PyTorch build does not support float8_e4m3fn")
        aq = a.to(torch.float8_e4m3fn)
        bq = b.to(torch.float8_e4m3fn)
        a_raw = aq.contiguous().view(torch.uint8).numpy().copy()
        b_raw = bq.contiguous().view(torch.uint8).numpy().copy()
        return aq.float(), bq.float(), a_raw, b_raw
    raise ValueError(f"unsupported dtype: {dtype}")


def prepare_case(elf: Path, shape: dict, args):
    """Generate kernel inputs and a PyTorch float32 golden result."""
    case_dir = COMPARE_ROOT / shape["name"]
    case_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    if args.ones:
        a = torch.ones((shape["M"], shape["K"]), dtype=torch.float32)
        b = torch.ones((shape["K"], shape["N"]), dtype=torch.float32)
    else:
        a = torch.randn((shape["M"], shape["K"]), generator=generator) * args.input_scale
        b = torch.randn((shape["K"], shape["N"]), generator=generator) * args.input_scale
        a.clamp_(-1.0, 1.0)
        b.clamp_(-1.0, 1.0)

    a_ref, b_ref, a_file, b_file = quantize_inputs(a, b, shape["dtype"])
    golden = torch.matmul(a_ref, b_ref).float().numpy()

    a_file.tofile(case_dir / "src0.bin")
    b_file.tofile(case_dir / "src1.bin")
    golden.astype(np.float32).tofile(case_dir / "golden.bin")
    np.zeros_like(golden, dtype=np.float32).tofile(case_dir / "res.bin")
    return case_dir, golden


def run_gfrun(elf: Path, args):
    command = [args.gfrun, *shlex.split(args.gfrun_args), str(elf)]
    print("command:", shlex.join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        return "timeout", -1, output
    return ("pass" if process.returncode == 0 else "fail"), process.returncode, output


def compare_result(case_dir: Path, golden: np.ndarray, args):
    result_path = case_dir / "res.bin"
    if not result_path.exists():
        return False, {"reason": "res.bin was not created"}

    result = np.fromfile(result_path, dtype=np.float32)
    if result.size != golden.size:
        return False, {
            "reason": f"shape mismatch: result elements={result.size}, golden={golden.size}"
        }
    result = result.reshape(golden.shape)
    diff = result - golden
    abs_diff = np.abs(diff)
    max_index = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    metrics = {
        "mse": float(np.mean(diff * diff)),
        "max_abs": float(abs_diff[max_index]),
        "max_index": tuple(int(i) for i in max_index),
        "actual_at_max": float(result[max_index]),
        "golden_at_max": float(golden[max_index]),
        "actual_min": float(np.min(result)),
        "actual_max": float(np.max(result)),
        "golden_min": float(np.min(golden)),
        "golden_max": float(np.max(golden)),
    }
    passed = bool(np.allclose(result, golden, atol=args.atol, rtol=args.rtol))

    report = case_dir / "golden_compare.log"
    with report.open("w", encoding="utf-8") as stream:
        stream.write(f"status: {'PASS' if passed else 'FAIL'}\n")
        stream.write(f"atol: {args.atol}\nrtol: {args.rtol}\n")
        for key, value in metrics.items():
            stream.write(f"{key}: {value}\n")
        stream.write("\nactual head:\n")
        stream.write(np.array2string(result[:4, :8], precision=8))
        stream.write("\n\ngolden head:\n")
        stream.write(np.array2string(golden[:4, :8], precision=8))
        stream.write("\n")
    metrics["report"] = str(report)
    return passed, metrics


def check_one(elf_text: str, args):
    elf = Path(elf_text).expanduser().resolve()
    if not elf.is_file():
        return False, {"elf": str(elf), "reason": "ELF does not exist"}
    try:
        shape = extract_shape(elf)
        case_dir, golden = prepare_case(elf, shape, args)
    except (ValueError, RuntimeError) as exc:
        return False, {"elf": str(elf), "reason": str(exc)}

    run_status, returncode, run_output = run_gfrun(elf, args)
    if run_status != "pass":
        run_log = case_dir / "gfrun.log"
        run_log.write_text(run_output, encoding="utf-8")
        return False, {
            "elf": str(elf),
            "shape": shape,
            "run_status": run_status,
            "returncode": returncode,
            "run_log": str(run_log),
        }

    passed, metrics = compare_result(case_dir, golden, args)
    return passed, {
        "elf": str(elf),
        "shape": shape,
        "run_status": run_status,
        "compare_status": "pass" if passed else "fail",
        "metrics": metrics,
    }


def collect_elfs(args):
    if args.elf:
        return [args.elf]
    with open(args.list, "r", encoding="utf-8") as stream:
        return [line.strip() for line in stream if line.strip() and not line.lstrip().startswith("#")]


def main():
    parser = argparse.ArgumentParser(
        description="Run matmul with gfrun and compare res.bin against PyTorch"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-d", "--elf", help="single matmul ELF")
    source.add_argument("-l", "--list", help="text file containing ELF paths")
    parser.add_argument("--ones", action="store_true", help="use deterministic all-one inputs")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gfrun", default=DEFAULT_GFRUN)
    parser.add_argument("--gfrun-args", default=DEFAULT_GFRUN_ARGS)
    args = parser.parse_args()

    elfs = collect_elfs(args)
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(check_one, elf, args) for elf in elfs]
        for future in as_completed(futures):
            passed, details = future.result()
            results.append(passed)
            print(f"{'PASS' if passed else 'FAIL'}: {details}")

    passed_count = sum(results)
    print(f"summary: pass={passed_count}, fail={len(results) - passed_count}")
    return 0 if results and all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
