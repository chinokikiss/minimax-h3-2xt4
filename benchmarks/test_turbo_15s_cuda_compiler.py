"""CPU-only guard tests for the scoped 15s Turbo CUDA13 Kaggle retry."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from patches.h3_operator_dependencies import (
    PTD_2NFE_CUDA13_TRIAL, TURBO_15S_CUDA13_TRIAL,
    compiler_qualification, configure_compiler,
)


class _Policy:
    dit_attention = "sol"
    row_linear_backend = "tilelang_scaled_fp16"
    column_linear_backend = "cuda_int8_tilelang"


class Turbo15sCompilerTests(unittest.TestCase):
    def test_default_rejects_unqualified_cuda13(self):
        with patch.dict(os.environ, {
            PTD_2NFE_CUDA13_TRIAL: "",
            TURBO_15S_CUDA13_TRIAL: "",
        }):
            with self.assertRaisesRegex(RuntimeError, "unqualified"):
                compiler_qualification(
                    "Cuda compilation tools, release 13.0, V13.0.88",
                    torch_cuda="13.0",
                )

    def test_validated_cuda128_never_requires_opt_in(self):
        with patch.dict(os.environ, {
            PTD_2NFE_CUDA13_TRIAL: "",
            TURBO_15S_CUDA13_TRIAL: "",
        }):
            self.assertEqual(
                compiler_qualification(
                    "Cuda compilation tools, release 12.8, V12.8.93",
                    torch_cuda="13.0",
                ),
                "validated_cuda_12_8",
            )

    def test_turbo_cuda130_opt_in_requires_matching_native_torch(self):
        with patch.dict(os.environ, {
            PTD_2NFE_CUDA13_TRIAL: "",
            TURBO_15S_CUDA13_TRIAL: "1",
        }):
            self.assertEqual(
                compiler_qualification(
                    "Cuda compilation tools, release 13.0, V13.0.88",
                    torch_cuda="13.0",
                ),
                "experimental_cuda_13_0_turbo_15s_only",
            )
            with self.assertRaisesRegex(RuntimeError, "torch_cuda=12.8"):
                compiler_qualification(
                    "Cuda compilation tools, release 13.0, V13.0.88",
                    torch_cuda="12.8",
                )

    def test_ptd_scope_remains_independent(self):
        with patch.dict(os.environ, {
            PTD_2NFE_CUDA13_TRIAL: "1",
            TURBO_15S_CUDA13_TRIAL: "",
        }):
            self.assertEqual(
                compiler_qualification(
                    "Cuda compilation tools, release 13.0, V13.0.88",
                    torch_cuda="13.0",
                ),
                "experimental_cuda_13_0_ptd_2nfe_only",
            )

    def test_no_other_cuda_release_admitted(self):
        with patch.dict(os.environ, {
            PTD_2NFE_CUDA13_TRIAL: "",
            TURBO_15S_CUDA13_TRIAL: "1",
        }):
            for release in ("11.8", "12.9", "13.1"):
                with self.subTest(release=release), self.assertRaises(RuntimeError):
                    compiler_qualification(
                        "Cuda compilation tools, release " + release,
                        torch_cuda="13.0",
                    )

    def test_selector_prefers_validated_cuda128(self):
        with TemporaryDirectory() as directory:
            trial = Path(directory) / "cuda13"
            validated = Path(directory) / "cuda128"
            trial.touch()
            validated.touch()

            def probe(argv, **_):
                if Path(argv[0]) == validated:
                    return "Cuda compilation tools, release 12.8, V12.8.93"
                return "Cuda compilation tools, release 13.0, V13.0.88"

            with patch.dict(os.environ, {
                PTD_2NFE_CUDA13_TRIAL: "",
                TURBO_15S_CUDA13_TRIAL: "1",
                "CUDACXX": str(trial),
            }), patch("shutil.which", return_value=str(validated)), \
                 patch("subprocess.check_output", side_effect=probe):
                info = configure_compiler(_Policy())
                self.assertEqual(info["qualification"], "validated_cuda_12_8")
                self.assertEqual(info["path"], str(validated.resolve()))

    def test_selector_turbo_fallback_cuda130(self):
        import torch
        with TemporaryDirectory() as directory:
            compiler = Path(directory) / "nvcc"
            compiler.touch()
            with patch.dict(os.environ, {
                PTD_2NFE_CUDA13_TRIAL: "",
                TURBO_15S_CUDA13_TRIAL: "1",
                "CUDACXX": str(compiler),
            }), patch("shutil.which", return_value=str(compiler)), \
                 patch("subprocess.check_output",
                       return_value="Cuda compilation tools, release 13.0, V13.0.88"), \
                 patch.object(torch.version, "cuda", "13.0"):
                info = configure_compiler(_Policy())
                self.assertEqual(
                    info["qualification"],
                    "experimental_cuda_13_0_turbo_15s_only",
                )
                self.assertEqual(os.environ["CUDACXX"], str(compiler.resolve()))


class SolNativeTurbo15sCompilerTests(unittest.TestCase):
    """Keep native Sol CUDA13 admission isolated to the existing Turbo flag."""

    def _probe_sol(self, *, release, turbo="", ptd="", torch_cuda="13.0"):
        from patches import sol_attention
        with TemporaryDirectory() as directory:
            compiler = Path(directory) / "nvcc"
            with patch.dict(os.environ, {
                "CUDACXX": str(compiler),
                TURBO_15S_CUDA13_TRIAL: turbo,
                PTD_2NFE_CUDA13_TRIAL: ptd,
            }), patch.object(sol_attention.subprocess, "check_output",
                          return_value="Cuda compilation tools, release " + release), \
                 patch.object(sol_attention.torch.version, "cuda", torch_cuda):
                return sol_attention._compiler()

    def test_sol_preserved_cuda128_remains_valid_without_opt_in(self):
        path, version = self._probe_sol(release="12.8")
        self.assertEqual(path.name, "nvcc")
        self.assertIn("release 12.8", version)

    def test_sol_rejects_cuda13_by_default(self):
        with self.assertRaisesRegex(RuntimeError, "preserved CUDA 12.8"):
            self._probe_sol(release="13.0")

    def test_sol_allows_scoped_turbo_cuda13_with_matching_torch(self):
        path, version = self._probe_sol(release="13.0", turbo="1")
        self.assertEqual(path.name, "nvcc")
        self.assertIn("release 13.0", version)

    def test_sol_rejects_ptd_only_cuda13_flag(self):
        with self.assertRaisesRegex(RuntimeError, "preserved CUDA 12.8"):
            self._probe_sol(release="13.0", ptd="1")

    def test_sol_rejects_turbo_cuda13_without_matching_torch(self):
        with self.assertRaisesRegex(RuntimeError, "native Torch cu130"):
            self._probe_sol(release="13.0", turbo="1", torch_cuda="12.8")

    def test_sol_rejects_other_toolkit_versions(self):
        with self.assertRaisesRegex(RuntimeError, "preserved CUDA 12.8"):
            self._probe_sol(release="13.1", turbo="1")

    def test_sol_preflight_precedes_weight_download(self):
        import inspect
        from benchmarks import run_h3_l1_turbo_batch
        body = inspect.getsource(run_h3_l1_turbo_batch.main)
        self.assertLess(body.index("sol_attention.prepare_native()"),
                        body.index("assets = json.loads"))


if __name__ == "__main__":
    unittest.main()
