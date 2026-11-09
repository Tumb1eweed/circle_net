"""Render the implemented CircleNet in a layout inspired by paper Fig. 5.

Red outlines mark modules added relative to the archived paper source. The
feature extractor follows this repository's two-branch implementation, rather
than the paper figure's illustrative four-branch extractor.
"""

from math import hypot
from pathlib import Path
from xml.sax.saxutils import escape


OUT = Path(__file__).resolve().parent
SVG = OUT / "point_attention_architecture.svg"
W, H = 1400, 840
INK = "#24313b"
BLUE = "#3979ad"
RED = "#ca3643"
LINE = "#687987"
parts = []


def add(s):
    parts.append(s)


def text(x, y, s, size=15, color=INK, anchor="middle", weight="normal"):
    add(f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        'font-family="DejaVu Sans, Noto Sans, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}">'
        f'{escape(s)}</text>')


def rect(x, y, w, h, fill="white", stroke="#aab9c4", radius=11, sw=2, dash=""):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra}/>')


def box(x, y, w, h, lines, *, new=False, fill=None, size=14):
    rect(x, y, w, h, fill or ("#fff1f2" if new else "white"),
         RED if new else "#aab9c4", sw=3 if new else 2)
    step = size + 7
    y0 = y + h / 2 - (len(lines) - 1) * step / 2 + size * .35
    for i, line in enumerate(lines):
        text(x + w / 2, y0 + i * step, line, size,
             weight="bold" if i == 0 else "normal")
    if new:
        rect(x + w - 48, y - 10, 42, 20, RED, RED, radius=4, sw=1)
        text(x + w - 27, y + 4, "NEW", 11, "white", weight="bold")


def path(points, *, red=False):
    color = RED if red else BLUE
    d = "M" + " L".join(f"{x} {y}" for x, y in points)
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="3" '
        'stroke-linejoin="round"/>')
    x0, y0 = points[-2]
    x1, y1 = points[-1]
    length = hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    bx, by = x1 - 10 * ux, y1 - 10 * uy
    add(f'<polygon points="{x1},{y1} {bx - 4 * uy},{by + 4 * ux} '
        f'{bx + 4 * uy},{by - 4 * ux}" fill="{color}"/>')


def panel(x, y, w, h, title):
    rect(x, y, w, h, "#f8fafc", "#81909b", radius=20)
    text(x + 20, y + 31, title, 18, anchor="start", weight="bold")


def main():
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="350mm" '
        f'height="210mm" viewBox="0 0 {W} {H}">')
    add(f'<rect width="{W}" height="{H}" fill="white"/>')
    text(W / 2, 35, "CircleNet with point-to-point attention", 25, weight="bold")
    text(W / 2, 62, "Figure 5-inspired layout · implemented mainline model · red frames = added modules", 15, LINE)

    # (a) Full network, arranged like the overview in the paper figure.
    rect(28, 82, 1344, 185, "#eaf3fa", INK, radius=24, sw=3, dash="10 7")
    text(52, 115, "(a) Full network", 19, anchor="start", weight="bold")
    top = [
        (56, 145, ("Input", "Q: 500 points", "C: 2000 points"), False),
        (237, 145, ("KNN", "K = 16 / 128"), False),
        (418, 155, ("Feature", "extraction"), False),
        (609, 150, ("Two-token", "fusion"), False),
        (795, 172, ("Point-to-point", "attention"), True),
        (1003, 145, ("Boundary", "+ weight heads"), False),
        (1184, 155, ("Plane + soft", "circle fit"), True),
    ]
    for x, width, lines, new in top:
        box(x, 142, width, 77, lines, new=new)
    for left, right in ((201, 237), (382, 418), (573, 609),
                        (759, 795), (967, 1003), (1148, 1184)):
        path(((left + 4, 180), (right - 7, 180)))
    text(1262, 244, "center + radius", 13, RED)

    # (b) Feature extraction actually implemented: two XYZ neighborhood paths.
    panel(28, 290, 408, 449, "(b) Feature extraction")
    box(56, 342, 352, 53, ("Q [500,3] + C [2000,3]",), fill="#eaf3fa")
    box(54, 435, 155, 68, ("Local patch", "K = 16"))
    box(255, 435, 155, 68, ("Wide patch", "K = 128"))
    path(((156, 395), (156, 415), (132, 415), (132, 435)))
    path(((308, 395), (308, 415), (332, 415), (332, 435)))
    box(54, 540, 155, 83, ("Conv1d ×3", "max pool", "128 channels"))
    box(255, 540, 155, 83, ("Conv1d ×3", "max pool", "128 channels"))
    path(((132, 503), (132, 540)))
    path(((332, 503), (332, 540)))
    box(122, 662, 220, 52, ("Two feature tokens / point",), fill="#eaf3fa")
    path(((132, 623), (132, 646), (210, 646), (210, 662)))
    path(((332, 623), (332, 646), (254, 646), (254, 662)))
    text(232, 731, "Shared across all 500 query points", 12, LINE)

    # (c) Existing within-point fusion, followed by new cross-point attention.
    panel(453, 290, 492, 449, "(c) Feature fusion and point attention")
    box(480, 342, 438, 62, ("Two-token Transformer · 1 layer, 4 heads",
                             "mean → F [500,128]"))
    path(((699, 404), (699, 444)))
    rect(474, 444, 450, 272, "none", RED, radius=16, sw=3)
    text(699, 472, "Cross-point module", 16, RED, weight="bold")
    box(495, 490, 185, 64, ("Position MLP", "XYZ → 128; add to F"), new=True, size=13)
    box(713, 490, 185, 64, ("Morton order", "sort 500 points"), new=True, size=13)
    path(((680, 522), (713, 522)), red=True)
    box(495, 580, 185, 64, ("Window attention", "32 points, 4 heads"), new=True, size=13)
    box(713, 580, 185, 64, ("Shifted windows", "shift = 16 points"), new=True, size=13)
    path(((805, 554), (805, 570), (587, 570), (587, 580)), red=True)
    path(((680, 612), (713, 612)), red=True)
    box(603, 667, 192, 39, ("Global attention · 500 points",), new=True, size=12)
    path(((805, 644), (805, 655), (699, 655), (699, 667)), red=True)

    # (d) Prediction heads and analytical fit; learned plane and soft fit are new.
    panel(962, 290, 410, 449, "(d) Heads and circle fitting")
    box(990, 342, 354, 53, ("Enriched point features [500,128]",), fill="#eaf3fa")
    box(985, 435, 112, 66, ("Boundary", "P [500,2]"), size=13)
    box(1111, 435, 112, 66, ("Weight", "W [500,1]"), size=13)
    box(1237, 435, 112, 66, ("Plane normal", "n [3]"), new=True, size=12)
    for x in (1041, 1167, 1293):
        path(((1167, 395), (1167, 418), (x, 418), (x, 435)))
    box(1006, 545, 326, 55, ("Soft fit weights",
                             "q = p² · (0.05 + 0.95 sigmoid(w))"), new=True, size=13)
    path(((1041, 501), (1041, 526), (1100, 526), (1100, 545)), red=True)
    path(((1167, 501), (1167, 526), (1220, 526), (1220, 545)), red=True)
    box(1006, 634, 326, 54, ("Differentiable weighted circle fit",
                             "Q + q + n → center c, radius r"), new=True, size=13)
    path(((1169, 600), (1169, 634)), red=True)
    path(((1293, 501), (1357, 501), (1357, 661), (1332, 661)), red=True)
    text(1167, 715, "One circle per positive patch", 13, LINE)

    rect(28, 761, 1344, 53, "#fafbfc", "#cbd4da", radius=8)
    rect(49, 780, 28, 16, "#fff1f2", RED, radius=3, sw=3)
    text(91, 793, "Added relative to the paper source implementation", 13, anchor="start")
    text(1326, 793, "Paper source: hard selection → PCA plane → weighted 2D least squares",
         12, LINE, anchor="end")
    add("</svg>")

    SVG.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(SVG)
    try:
        import pymupdf
    except ImportError:
        print("PyMuPDF unavailable: SVG saved; PNG/PDF export skipped")
    else:
        document = pymupdf.open(SVG)
        scale = 2800 / document[0].rect.width
        document[0].get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                               alpha=False).save(OUT / "point_attention_architecture.png")
        (OUT / "point_attention_architecture.pdf").write_bytes(document.convert_to_pdf())


if __name__ == "__main__":
    main()
