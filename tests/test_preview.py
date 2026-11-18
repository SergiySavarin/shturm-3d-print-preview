"""CLI integration checks with tiny fixtures; run with unittest discover."""
import json
import io
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile

from PIL import Image

from pypdf import PdfReader

SCRIPT = Path(__file__).resolve().parents[1] / "preview.py"


class PreviewTests(unittest.TestCase):
    def test_cli_success_partial_failure_and_recursive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            nested = source / "nested"
            nested.mkdir()
            vertices = [(0, 0, 0), (10, 0, 0), (0, 20, 0), (0, 0, 30)]
            faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
            data = b"test".ljust(80, b"\0") + struct.pack("<I", len(faces))
            for face in faces:
                data += struct.pack("<12fH", 0, 0, 0, *(v for i in face for v in vertices[i]), 0)
            (nested / "tetra.stl").write_bytes(data)
            out = root / "result.pdf"
            cad = root / "result-step-stl.pdf"
            three_mf = root / "result-3mf.pdf"

            def run(*extra):
                return subprocess.run([sys.executable, str(SCRIPT), str(source), "-o", str(out), *extra], capture_output=True, text=True)

            self.assertEqual(run().returncode, 1)  # no supported files at this level
            self.assertFalse(out.exists())
            result = run("--recursive")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(out.exists())
            self.assertFalse(three_mf.exists())
            report = json.loads(cad.with_suffix(".json").read_text())
            self.assertEqual(report["files"][0]["dimensions_mm"], [10, 20, 30])
            self.assertEqual(len(PdfReader(cad).pages), 2)
            self.assertNotIn(" KB", PdfReader(cad).pages[0].extract_text())
            (source / "broken.step").write_text("invalid STEP")
            result = run("--recursive")
            self.assertEqual(result.returncode, 2, result.stderr)
            report = json.loads(cad.with_suffix(".json").read_text())
            self.assertEqual([e["status"] for e in report["files"]], ["error", "ok"])
            self.assertEqual(len(PdfReader(cad).pages), 3)
            self.assertIn("broken.step", PdfReader(cad).pages[1].extract_text())
            self.assertEqual((nested / "tetra.stl").read_bytes(), data)

            image = io.BytesIO()
            Image.new("RGB", (64, 64), "orange").save(image, format="PNG")
            for name in ("first.3mf", "second.3MF"):
                with zipfile.ZipFile(source / name, "w") as archive:
                    archive.writestr("Metadata/plate_1.png", image.getvalue())
                    archive.writestr("Metadata/plate_2.png", image.getvalue())
            result = run("--recursive")
            self.assertEqual(result.returncode, 2, result.stderr)
            mf_text = "\n".join(page.extract_text() for page in PdfReader(three_mf).pages)
            cad_text = "\n".join(page.extract_text() for page in PdfReader(cad).pages)
            mf_pages = PdfReader(three_mf).pages
            self.assertEqual(len(mf_pages), 5)
            self.assertNotIn(" KB", mf_pages[0].extract_text())
            for page, number in zip(mf_pages[1:], (1, 2, 1, 2)):
                self.assertIn(f"Пластина {number}", page.extract_text())
                self.assertEqual(len(page.images), 1)
            self.assertIn("first.3mf", mf_text)
            self.assertIn("second.3MF", mf_text)
            self.assertNotIn("tetra.stl", mf_text)
            self.assertNotIn("first.3mf", cad_text)
            self.assertEqual(len(json.loads(three_mf.with_suffix(".json").read_text())["files"]), 2)

            # Non-recursive input now has only 3MF plus a broken STEP.
            # A separate output base verifies 3MF-only creates no CAD document.
            only = root / "only"
            only.mkdir()
            (only / "first.3mf").write_bytes((source / "first.3mf").read_bytes())
            result = subprocess.run([sys.executable, str(SCRIPT), str(only), "-o", str(root / "only-preview.pdf")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "only-preview-3mf.pdf").exists())
            self.assertFalse((root / "only-preview-step-stl.pdf").exists())


if __name__ == "__main__":
    unittest.main()
