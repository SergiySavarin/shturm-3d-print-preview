#!/usr/bin/env python3
"""Generate a read-only CAD folder catalog. No network or Drive credentials needed."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile
from datetime import datetime
from xml.sax.saxutils import escape

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).parent / ".cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph

SUPPORTED = {".step", ".stp", ".stl", ".3mf"}


def mesh(path):
    from OCP.STEPControl import STEPControl_Reader
    from OCP.StlAPI import StlAPI_Reader
    from OCP.TopoDS import TopoDS, TopoDS_Shape
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    if path.suffix.lower() == ".stl":
        shape = TopoDS_Shape()
        if not StlAPI_Reader().Read(shape, str(path)):
            raise ValueError("Cannot read STL")
    else:
        reader = STEPControl_Reader()
        if reader.ReadFile(str(path)) != IFSelect_RetDone:
            raise ValueError("Cannot read STEP")
        if reader.TransferRoots() == 0:
            raise ValueError("No STEP shapes")
        shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, 0.15, False, 0.3, True).Perform()
    triangles = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            transform = loc.Transformation()
            for i in range(1, tri.NbTriangles() + 1):
                triangles.append([[p.X(), p.Y(), p.Z()] for p in
                                  [tri.Node(j).Transformed(transform) for j in tri.Triangle(i).Get()]])
        exp.Next()
    if not triangles:
        raise ValueError("No triangulated faces")
    return np.asarray(triangles)


def render(triangles, target):
    lo, hi = triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))
    center = (hi + lo) / 2
    radius = max(float((hi - lo).max()) / 2, 0.01) * 1.08
    fig = plt.figure(figsize=(10, 6), facecolor="#f3f6fa")
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    light = np.array([0.3, -0.5, 0.8])
    shade = 0.45 + 0.5 * np.abs(normals @ light)
    colors = np.column_stack([shade * 0.25, shade * 0.58, shade * 0.8, np.ones(len(shade))])
    for index, (elev, azim, label) in enumerate([(25, -55, "Isometric"), (90, -90, "Top / XY"), (0, -90, "Front / XZ")]):
        ax = fig.add_subplot(2, 2, (1, 3) if index == 0 else index * 2, projection="3d")
        ax.add_collection3d(Poly3DCollection(triangles, facecolors=colors, linewidths=0, rasterized=True))
        ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius), zlim=(center[2]-radius, center[2]+radius))
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type("ortho")
        ax.set_axis_off()
        ax.set_facecolor("#f3f6fa")
        ax.set_title(label, fontsize=10, color="#526277")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=.94, wspace=0, hspace=.05)
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return (hi-lo).tolist()


def fonts(directory=None):
    roots = [Path(directory)] if directory else [Path("/System/Library/Fonts/Supplemental"), Path("/usr/share/fonts/truetype/dejavu")]
    for root in roots:
        for regular, bold in [("Arial.ttf", "Arial Bold.ttf"), ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")]:
            if (root / regular).is_file() and (root / bold).is_file():
                pdfmetrics.registerFont(TTFont("Body", str(root / regular)))
                pdfmetrics.registerFont(TTFont("Bold", str(root / bold)))
                return
    raise ValueError("Unicode fonts missing; use --font-dir with Arial or DejaVuSans")


def _generate_catalog(source, output, files, kind, font_dir=None):
    source, output = source.resolve(), output.resolve()
    fonts(font_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with tempfile.TemporaryDirectory(prefix="cad-preview-") as temp:
        temp = Path(temp)
        for index, path in enumerate(files):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            entry = {"file": str(path.relative_to(source)), "bytes": path.stat().st_size,
                     "sha256": digest}
            try:
                if path.suffix.lower() == ".3mf":
                    with zipfile.ZipFile(path) as archive:
                        names = sorted((n for n in archive.namelist() if n.startswith("Metadata/plate_") and n.removeprefix("Metadata/plate_").removesuffix(".png").isdigit()), key=lambda n: int(n.split("_")[-1].split(".")[0]))
                        entry["images"] = [archive.read(n) for n in names]
                        if not names:
                            raise ValueError("3MF has no embedded plate previews; geometry fallback is not implemented")
                else:
                    image = temp / f"{index}.png"
                    entry["dimensions_mm"] = render(mesh(path), image)
                    entry["images"] = [image.read_bytes()]
                entry["status"] = "ok"
            except Exception as exc:
                entry.update(status="error", error=str(exc))
            entries.append(entry)
            print(f"{entry['status']}: {entry['file']}", flush=True)

        c = canvas.Canvas(str(temp / "catalog.pdf"), pagesize=(595.28, 841.89))
        title = "3MF · Прев’ю" if kind == "3mf" else "STEP / STL · Прев’ю"
        c.setTitle(f"{source.name} - {title}")
        page = 0

        def text(value, y, size=11, bold=False, color="#203348", x=40, width=515):
            style = ParagraphStyle("text", fontName="Bold" if bold else "Body", fontSize=size, leading=size*1.4, textColor=color)
            p = Paragraph(escape(str(value)), style)
            _, height = p.wrap(width, 700)
            p.drawOn(c, x, y-height)
            return y-height

        def new_page(label):
            nonlocal page
            if page:
                c.showPage()
            page += 1
            c.setFillColorRGB(.07, .16, .24)
            c.rect(0, 770, 596, 72, fill=1, stroke=0)
            text(label, 813, 17, True, "#ffffff")
            text(f"{source.name}  |  {page:02d}", 28, 9, color="#6a7787")

        new_page(title)
        y = text(source.name, 738, 27, True)
        y = text(f"Згенеровано: {datetime.now().astimezone().isoformat(timespec='minutes')}", y-12, 10)
        y = text(f"Файлів: {len(entries)} · Успішно: {sum(e['status']=='ok' for e in entries)} · Помилок: {sum(e['status']=='error' for e in entries)}", y-12)
        description = ("3MF: вбудовані зображення пластин, які можуть відрізнятися від поточного стану геометрії."
                       if kind == "3mf" else
                       "STEP: геометричні прев’ю та осьові габарити в мм. STL: одиниці умовно прийняті за мм. Це візуальний каталог, не креслення.")
        y = text(description, y-18, 10)
        y = text("Вміст папки", y-24, 15, True)
        for entry in entries:
            if y < 110:
                new_page("Вміст папки")
                y = 735
            y = text(entry["file"], y-14, 10, True)
            y = text(entry['status'].upper(), y-3, 9, color="#61758a")
        for entry in entries:
            if entry["status"] == "error":
                new_page("Не вдалося обробити")
                y = text(entry["file"], 735, 15, True)
                text(entry["error"], y-18)
                continue
            images = entry["images"]
            for start in range(len(images)):
                new_page("3MF · Прев’ю пластин" if entry["file"].lower().endswith(".3mf") else "Деталь · Геометричне прев’ю")
                y = text(entry["file"], 738, 15, True)
                if "dimensions_mm" in entry:
                    text("Габарити X × Y × Z: " + " × ".join(f"{v:.2f}" for v in entry["dimensions_mm"]) + " мм", y-12)
                    c.drawImage(ImageReader(io.BytesIO(images[0])), 35, 245, width=525, height=370, preserveAspectRatio=True, anchor="c")
                    text("Орієнтація з вихідного файлу. Масштаб автоматично підібрано для кожної деталі.", 190, 10)
                else:
                    text("Вбудовані зображення з 3MF; без повторного рендерингу геометрії.", y-12, 10)
                    text(f"Пластина {start+1}", y-43, 16, True)
                    c.drawImage(ImageReader(io.BytesIO(images[start])), 40, 115,
                                width=515, height=505, preserveAspectRatio=True,
                                anchor="c", mask="auto")
        c.save()
        os.replace(temp / "catalog.pdf", output)
    report = {"source": str(source), "output": str(output), "pages": page,
              "files": [{k: v for k, v in e.items() if k != "images"} for e in entries]}
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def generate(source, output, recursive=False, font_dir=None):
    """Treat output as a filename base; emit one catalog per nonempty format group."""
    source, output = source.resolve(), output.resolve()
    if not source.is_dir():
        raise ValueError(f"Not a directory: {source}")
    files = sorted((p for p in (source.rglob("*") if recursive else source.iterdir())
                    if p.is_file() and p.suffix.lower() in SUPPORTED),
                   key=lambda p: str(p.relative_to(source)).casefold())
    if not files:
        raise ValueError("No supported STEP/STP/STL/3MF files found")
    reports = []
    base = output.with_suffix("") if output.suffix.lower() == ".pdf" else output
    for kind in ("3mf", "step-stl"):
        group = [p for p in files if (p.suffix.lower() == ".3mf") == (kind == "3mf")]
        if group:
            target = base.with_name(f"{base.name}-{kind}.pdf")
            reports.append(_generate_catalog(source, target, group, kind, font_dir))
    return {"outputs": reports, "files": [entry for report in reports for entry in report["files"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("output/pdf/preview.pdf"),
                        help="Filename base; creates <base>-3mf.pdf and <base>-step-stl.pdf when present")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--font-dir")
    args = parser.parse_args()
    try:
        report = generate(args.folder, args.output, args.recursive, args.font_dir)
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")
    for result in report["outputs"]:
        print(f"PDF: {result['output']} ({result['pages']} pages)")
    return 2 if any(e["status"] == "error" for e in report["files"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
