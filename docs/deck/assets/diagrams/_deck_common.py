"""Shared SVG helpers for the AIVC appendix deck (Batch 2 onwards: B1-E1).

A1-A4 builders are self-contained (they shipped before this module existed).
B1-E1 builders import from here for DRY palette / typography / header / footer
/ background / lock icon.

Style source of truth:
- docs/deck/assets/color_palette.md
- docs/deck/assets/typography.md
- docs/deck/assets/icon_inventory.md

All builders share:
- 1920×1080 viewBox
- Dark background #070A14 with corner radial glows (cyan + purple)
- APPENDIX <ID> · <SECTION> eyebrow at y=78
- Title at y=138, subtitle at y=186
- Footer divider at y=H-132, text + pagination at y=H-100
"""
from __future__ import annotations

# ===== Palette (locked) =====
BG          = "#070A14"
SURFACE     = "#0F1428"
SURFACE_2   = "#0B1020"
SURFACE_REJ = "#10141F"

CYAN        = "#26DDF9"
CYAN_HI     = "#00F2FF"
PURPLE      = "#8B5CF6"
LAVENDER    = "#B47DF0"

TEXT_TITLE  = "#F7FAFF"
TEXT_BODY   = "#EAF6FF"
TEXT_MUTED  = "#A8B4C2"
TEXT_DIM    = "#94A3B8"
TEXT_DISABLED = "#5B6478"

OK_GREEN    = "#4ADE80"
WARN_AMBER  = "#FBBF24"
DANGER_RED  = "#FF4D6D"
DIVIDER     = "#1A2235"

# Section accent rotation (from Batch 2 prompt)
SECTION_ACCENT = {
    "A": (CYAN,     PURPLE),       # already locked across A1-A4
    "B": (OK_GREEN, CYAN),         # validation
    "C": (CYAN,     PURPLE),       # QurieSeq — cyan = the moat
    "D": (LAVENDER, TEXT_BODY),    # roadmap + budget
    "E": (TEXT_BODY, CYAN),        # horizon — pale gradient
}

# ===== Layout constants =====
W, H = 1920, 1080
START_X = 96

# ===== Typography =====
FONT = "Inter, -apple-system, 'Helvetica Neue', Arial, sans-serif"
FONT_BODY = "Arial, Inter, 'Helvetica Neue', sans-serif"
FONT_MATH = "Inter, 'Cambria Math', 'STIX Two Math', serif"
FONT_MONO = "'SF Mono', Menlo, Consolas, monospace"


# ===== SVG primitives =====
def svg_open(aria_label: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-label="{aria_label}">'
    )


def background(parts: list[str], extra_defs: str = "") -> None:
    """Standard dark background + two corner radial glows."""
    parts.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    parts.append(
        '<defs>'
        f'<radialGradient id="glow1" cx="0.85" cy="0.15" r="0.55">'
        f'<stop offset="0%" stop-color="{CYAN}" stop-opacity="0.10"/>'
        f'<stop offset="100%" stop-color="{CYAN}" stop-opacity="0"/>'
        f'</radialGradient>'
        f'<radialGradient id="glow2" cx="0.15" cy="0.95" r="0.5">'
        f'<stop offset="0%" stop-color="{PURPLE}" stop-opacity="0.10"/>'
        f'<stop offset="100%" stop-color="{PURPLE}" stop-opacity="0"/>'
        f'</radialGradient>'
        f'{extra_defs}'
        '</defs>'
        f'<rect width="{W}" height="{H}" fill="url(#glow1)"/>'
        f'<rect width="{W}" height="{H}" fill="url(#glow2)"/>'
    )


def header(parts: list[str], appendix_id: str, section: str, title: str,
           subtitle: str, eyebrow_color: str = CYAN) -> None:
    """Standard appendix header: eyebrow + divider + title + subtitle."""
    parts.append(
        f'<text x="{START_X}" y="78" fill="{eyebrow_color}" font-family="{FONT}" '
        f'font-size="14" font-weight="700" letter-spacing="4">'
        f'APPENDIX {appendix_id} · {section}</text>'
    )
    parts.append(
        f'<line x1="{START_X + 380}" y1="72" x2="{START_X + 600}" y2="72" '
        f'stroke="{eyebrow_color}" stroke-opacity="0.4" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{START_X}" y="138" fill="{TEXT_TITLE}" font-family="{FONT}" '
        f'font-size="40" font-weight="700" letter-spacing="-0.5">{title}</text>'
    )
    parts.append(
        f'<text x="{START_X}" y="186" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="18" font-weight="400">{subtitle}</text>'
    )


def footer(parts: list[str], source_text: str, slide_handle: str,
           handle_color: str = CYAN) -> None:
    """Standard footer divider + citation + pagination."""
    fy_line = H - 132
    fy_text = H - 100
    parts.append(
        f'<line x1="{START_X}" y1="{fy_line}" x2="{W - START_X}" y2="{fy_line}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{START_X}" y="{fy_text}" fill="{TEXT_MUTED}" font-family="{FONT_BODY}" '
        f'font-size="11" font-weight="400" font-style="italic">{source_text}</text>'
    )
    parts.append(
        f'<text x="{W - START_X}" y="{fy_text}" fill="{handle_color}" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="2" text-anchor="end">{slide_handle}</text>'
    )


def lock_icon(cx: int, cy: int, color: str, scale: float = 1.0) -> str:
    """Inline padlock — 16x18 px at scale 1.0, centered on (cx, cy)."""
    s = scale
    return (
        f'<g transform="translate({cx-8*s},{cy-9*s}) scale({s})">'
        f'<rect x="0" y="6" width="16" height="12" rx="2" fill="{color}" '
        f'fill-opacity="0.18" stroke="{color}" stroke-width="1.6"/>'
        f'<path d="M3 6 V3.5 A4.5 4.5 0 0 1 13 3.5 V6" fill="none" '
        f'stroke="{color}" stroke-width="1.6" stroke-linecap="round"/>'
        f'<circle cx="8" cy="12" r="1.2" fill="{color}"/>'
        f'</g>'
    )


def section_eyebrow(parts: list[str], x: int, y: int, label: str,
                    color: str = TEXT_MUTED, divider_w: int = 0) -> None:
    """Letter-spaced bold caps section label (10-12pt) used inside zones."""
    parts.append(
        f'<text x="{x}" y="{y}" fill="{color}" font-family="{FONT}" '
        f'font-size="12" font-weight="700" letter-spacing="3">{label}</text>'
    )
    if divider_w > 0:
        # caller can pass divider_w to extend a thin rule to the right
        # of the eyebrow text
        approx_text_w = len(label) * 8  # rough visible-char width estimate
        x1 = x + approx_text_w + 20
        parts.append(
            f'<line x1="{x1}" y1="{y-6}" x2="{x + divider_w}" y2="{y-6}" '
            f'stroke="{DIVIDER}" stroke-width="1"/>'
        )


def arrow(parts: list[str], x1: int, y1: int, x2: int, y2: int,
          color: str = CYAN, opacity: float = 0.55, width: float = 2.0) -> None:
    """Thin line + chevron arrow head, default cyan."""
    parts.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2-6}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{width}" stroke-opacity="{opacity}" stroke-linecap="round"/>'
    )
    parts.append(
        f'<path d="M {x2-8} {y2-5} L {x2} {y2} L {x2-8} {y2+5}" fill="none" '
        f'stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity+0.2}" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def render_png(svg_path, png_path):
    """Render a finalized SVG file to PNG via cairosvg (1920×1080)."""
    import cairosvg, pathlib
    src = pathlib.Path(svg_path)
    cairosvg.svg2png(
        bytestring=src.read_bytes(),
        write_to=str(png_path),
        output_width=W, output_height=H,
    )
