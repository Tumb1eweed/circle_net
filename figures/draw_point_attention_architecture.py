"""Draw the trained CircleNetPointAttention architecture as an editable SVG.

Run with the circlenet-repro environment. CairoSVG is used only to produce the
PNG/PDF companions; the SVG is generated with the Python standard library.
"""

from pathlib import Path
from xml.sax.saxutils import escape


OUT = Path(__file__).resolve().parent
SVG = OUT / "point_attention_architecture.svg"
W, H = 519, 706  # 183 mm wide at one SVG unit per typographic point
INK = "#1c2d3b"
LINE = "#65798b"
ACCENT = "#2c6096"
NEUTRAL = "#f4f6f8"
PALE = "#eaf2fa"
FONT = "DejaVu Sans, WenQuanYi Zen Hei, sans-serif"
parts = []


def add(value):
    parts.append(value)


def text(x, y, value, size=6.7, anchor="middle", weight="normal", fill=INK):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}">{escape(value)}</text>')


def box(ident, x, y, width, height, lines, *, accent=False, title=False):
    fill = PALE if accent else NEUTRAL
    stroke = ACCENT if accent else "#b7c3cc"
    add(f'<g id="{ident}">')
    add(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
    spacing = 11.1
    first = y + (height - spacing * (len(lines) - 1)) / 2 + 2.4
    for index, value in enumerate(lines):
        add(text(x + width / 2, first + index * spacing, value,
                 size=7.0 if index == 0 and title else 6.65,
                 weight="bold" if index == 0 and title else "normal"))
    add("</g>")


def arrow(ident, d, *, accent=False, width=1.25, head=True):
    stroke = ACCENT if accent else LINE
    marker = ' marker-end="url(#arrow-accent)"' if accent and head else (
        ' marker-end="url(#arrow)"' if head else "")
    add(f'<path id="{ident}" d="{d}" fill="none" stroke="{stroke}" '
        f'stroke-width="{width}" stroke-linejoin="round"{marker}/>')


def dot(x, y):
    add(f'<circle cx="{x}" cy="{y}" r="2.2" fill="{LINE}"/>')


def main():
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="183mm" '
        f'height="249mm" viewBox="0 0 {W} {H}">')
    add("<defs>")
    for ident, fill in (("arrow", LINE), ("arrow-accent", ACCENT)):
        add(f'<marker id="{ident}" markerWidth="7" markerHeight="7" '
            'refX="6" refY="3.5" orient="auto" markerUnits="strokeWidth">'
            f'<path d="M0,0 L7,3.5 L0,7 Z" fill="{fill}"/></marker>')
    add("</defs>")
    add(f'<rect width="{W}" height="{H}" fill="white"/>')
    add(text(259.5, 20, "CircleNet point-attention architecture", 10.1,
             weight="bold"))
    add(text(259.5, 35, "Current trained branch  ·  B = batch size  ·  one target circle per l0 patch", 6.8))

    # Connectors are drawn first so that nodes cover their endpoints.
    arrow("e-input-query-knn", "M128 99 V111 H210 V121")
    arrow("e-input-context-knn", "M392 99 V111 H310 V121")
    arrow("e-knn-local", "M210 175 V185 H133 V198")
    arrow("e-knn-context", "M310 175 V185 H386 V198")
    arrow("e-local-fusion", "M133 256 V267 H181 V279")
    arrow("e-context-fusion", "M386 256 V267 H338 V279")
    arrow("e-fusion-attention", "M260 338 V349 H69 V388", accent=True, width=1.55)
    arrow("e-query-position", "M25 75 H9 V421 H25", width=1.05)
    arrow("e-query-fit", "M9 421 V649 H100", width=1.05)
    dot(9, 421)
    for ident, x1, x2 in (("e-pos-morton", 113, 122),
                           ("e-morton-local", 210, 219),
                           ("e-local-shift", 307, 316),
                           ("e-shift-global", 404, 413)):
        arrow(ident, f"M{x1} 423 H{x2}", accent=True, width=1.55)
    arrow("e-global-head-trunk", "M457 458 V483 H98", width=1.35, head=False)
    for x in (98, 260, 421):
        dot(x, 483)
        arrow(f"e-feature-head-{x}", f"M{x} 483 V500")
    arrow("e-class-weight", "M98 553 V563 H170 V573")
    arrow("e-head-weight", "M260 553 V573")
    arrow("e-weights-fit", "M260 612 V629")
    arrow("e-normal-fit", "M421 553 H511 V649 H419")

    box("n-query", 25, 52, 205, 47,
        ("Query patch · data_l0", "Q  [B, 500, 3]  XYZ"), title=True)
    box("n-context", 289, 52, 205, 47,
        ("Context patch · data_l1", "C  [B, 2000, 3]  XYZ"), title=True)
    box("n-knn", 125, 121, 269, 54,
        ("Normalize + KNN in C", "128 nearest neighbors for each Q point",
         "relative XYZ  [B, 500, 128, 3]"), title=True)
    box("n-local-encoder", 25, 198, 215, 58,
        ("Local neighborhood encoder · K=16", "Conv1d  3 → 64 → 128 → 128; max over K",
         "f_local  [B, 500, 128]"), title=True)
    box("n-context-encoder", 279, 198, 215, 58,
        ("Context neighborhood encoder · K=128", "Conv1d  3 → 64 → 128 → 128; max over K",
         "f_context  [B, 500, 128]"), title=True)
    box("n-two-token-fusion", 96, 279, 327, 59,
        ("Original two-token fusion, independently per Q point",
         "stack  [B×500, 2, 128]  →  Transformer 1 layer, 4 heads",
         "mean over 2 tokens  →  F  [B, 500, 128]"), title=True)

    add('<g id="c-point-attention">')
    add('<rect x="16" y="355" width="487" height="121" rx="7" fill="none" '
        f'stroke="{ACCENT}" stroke-width="1.35"/>')
    add(text(259.5, 371, "NEW  ·  attention between l0 points", 7.2,
             weight="bold", fill=ACCENT))
    add("</g>")
    box("n-position", 25, 388, 88, 70,
        ("Pos MLP + add", "Q: 3 → 64 → 128", "F+P [B,500,128]"), accent=True)
    box("n-morton", 122, 388, 88, 70,
        ("Morton sort", "Z-order; N=500", "[B,500,128]"), accent=True)
    box("n-local-attn", 219, 388, 88, 70,
        ("Window MHA", "32 pts; 4 heads", "[B×16,32,128]*"), accent=True)
    box("n-shifted-attn", 316, 388, 88, 70,
        ("Shifted MHA", "shift=16 pts", "32 pts; 4 heads",
         "[B×16,32,128]*"), accent=True)
    box("n-global-attn", 413, 388, 88, 70,
        ("Global MHA + LN", "500 pts; 4 heads", "[B,500,128]"), accent=True)

    box("n-classifier", 25, 500, 147, 53,
        ("Boundary classifier", "MLP 128 → 64 → 2", "logits [B,500,2]"), title=True)
    box("n-weight-head", 187, 500, 147, 53,
        ("Point-weight head", "MLP 128 → 64 → 1", "w logits [B,500]"), title=True)
    box("n-normal-head", 349, 500, 145, 53,
        ("Plane-normal head", "mean pool; 128 → 64 → 3", "n [B,3]"), title=True)
    box("n-soft-weights", 125, 573, 269, 39,
        ("Soft fit weights  q = p²(0.05 + 0.95 σ(w))",
         "p = softmax(logits)[...,1];  q [B,500]"), title=True)
    box("n-circle-fit", 100, 629, 319, 49,
        ("Differentiable weighted 3D circle fit on Q, q and n",
         "center c [B,3]   ·   radius r [B]   ·   normal n [B,3]"), title=True)
    add(text(259.5, 691, "* 500 points are padded to 512 for 16 windows; 12 padding tokens are masked.", 5.6))
    add(text(259.5, 701, "Each blue MHA block: LN → MHA → residual → LN → MLP(128→256→128) → residual.", 5.6))
    add("</svg>")
    OUT.mkdir(parents=True, exist_ok=True)
    SVG.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(SVG)
    try:
        import cairosvg
    except ImportError:
        print("CairoSVG unavailable: SVG saved; PNG/PDF export skipped")
    else:
        cairosvg.svg2png(url=str(SVG), write_to=str(OUT / "point_attention_architecture.png"),
                         output_width=2076, output_height=2824)
        cairosvg.svg2pdf(url=str(SVG), write_to=str(OUT / "point_attention_architecture.pdf"))
        print(OUT / "point_attention_architecture.png")
        print(OUT / "point_attention_architecture.pdf")


if __name__ == "__main__":
    main()
