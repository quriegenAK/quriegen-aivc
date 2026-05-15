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


# =============================================================================
# Collision-guard helpers (added Batch 2 fixes, 2026-05-15)
#
# The Batch 2 ship sweep landed 3 same-class bugs: B2 / D1 / D2 each had two
# text elements positioned at conflicting coordinates, rendering on top of
# each other. Textual acceptance checks (grep) couldn't catch these because
# both colliding strings are technically "present in the SVG".
#
# These heuristics are FIRST-LINE smoke tests, not proofs. They flag the
# common case (two short text labels sharing an x/y band). Edge cases
# they DON'T catch:
#   - text inside <g transform="..."> with translation/rotation
#   - text-anchor="end" / "middle" with long strings (we approximate)
#   - multi-line text via <tspan> (we treat the parent <text> y only)
#   - text rendered partially outside its parent rect by tspan offsets
# Visual review remains the authoritative final check.
# =============================================================================
import re as _re


def _extract_text_elements(svg_xml: str) -> list[dict]:
    """Extract <text> elements with x/y/font-size/text-anchor/content.

    Uses regex (not full XML parsing) because cairosvg's SVG output is
    well-formed but the parser context is irrelevant for this purpose.
    Returns one dict per <text> opening tag found.
    """
    out: list[dict] = []
    # Match the outer <text ...> tag and the text content up to </text>
    pattern = _re.compile(
        r'<text\b([^>]*)>(.*?)</text>',
        _re.DOTALL,
    )
    for m in pattern.finditer(svg_xml):
        attrs_str = m.group(1)
        content_raw = m.group(2)
        # Strip inner tspan markup to get the visible char count
        content_visible = _re.sub(r'<[^>]+>', '', content_raw)

        def attr(name, default=None, cast=str):
            mm = _re.search(rf'\b{name}="([^"]*)"', attrs_str)
            if not mm:
                return default
            try:
                return cast(mm.group(1))
            except (ValueError, TypeError):
                return default

        x = attr("x", 0.0, float)
        y = attr("y", 0.0, float)
        font_size = attr("font-size", 12.0, float)
        text_anchor = attr("text-anchor", "start")

        # Estimate effective horizontal extent.
        # Heuristic: average glyph width ≈ font_size * 0.65 for sans-serif.
        # (slightly aggressive to catch tight-but-overlapping labels —
        # tuned against known Batch 2 B2/D1 collisions; Inter actual metrics
        # render slightly wider than 0.55-0.6 estimates.)
        n_chars = len(content_visible.strip())
        width = n_chars * font_size * 0.65
        if text_anchor == "end":
            x_left = x - width
            x_right = x
        elif text_anchor == "middle":
            x_left = x - width / 2
            x_right = x + width / 2
        else:  # start
            x_left = x
            x_right = x + width

        # Vertical extent estimate (text baseline is at y; ascent ~80% above)
        y_top = y - font_size * 0.85
        y_bot = y + font_size * 0.15

        out.append({
            "x": x, "y": y,
            "x_left": x_left, "x_right": x_right,
            "y_top": y_top, "y_bot": y_bot,
            "font_size": font_size,
            "text_anchor": text_anchor,
            "content": content_visible.strip(),
        })
    return out


def check_no_text_collisions(svg_xml: str, *, min_gap: int = 4) -> list[tuple]:
    """Scan SVG <text> elements and flag probable rendering collisions.

    Algorithm: two text elements are flagged as colliding if their estimated
    bounding boxes (in CSS px) overlap by more than `min_gap` on BOTH axes.
    Bounding box is computed from x/y baseline + text-anchor adjustment +
    visible-char-width estimate (font_size * 0.6) + vertical ascent/descent
    (font_size * 0.85 / 0.15).

    Returns a list of tuples:
        (content1, content2, x_overlap_px, y_overlap_px)
    Empty list = no suspected collisions.

    Heuristic limits (per module docstring):
      - Glyph width approximated as font_size * 0.6 (sans-serif avg)
      - text-anchor end/middle adjusted; start assumed otherwise
      - Doesn't handle <g transform> translations

    Use as a pre-write smoke test in build scripts, not as authoritative
    visual review.
    """
    elems = _extract_text_elements(svg_xml)
    collisions: list[tuple] = []
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            a, b = elems[i], elems[j]
            # Skip empty content
            if not a["content"] or not b["content"]:
                continue
            # Two-axis bounding-box overlap
            x_overlap = min(a["x_right"], b["x_right"]) - max(a["x_left"], b["x_left"])
            y_overlap = min(a["y_bot"], b["y_bot"]) - max(a["y_top"], b["y_top"])
            if x_overlap > min_gap and y_overlap > min_gap:
                collisions.append((
                    a["content"][:60], b["content"][:60],
                    round(x_overlap, 1), round(y_overlap, 1),
                ))
    return collisions


def check_text_within_bounds(svg_xml: str, *, parent_bounds: list,
                              tolerance: int = 8) -> list[tuple]:
    """Verify every <text> element's estimated bbox sits inside at least one
    parent rect from `parent_bounds`.

    `parent_bounds` is a list of (x, y, w, h) tuples — usually the card/zone
    rectangles a builder draws. Catches off-card text (the A3 v1 Δ_synergy
    bug class) and other "text rendered outside its parent card" issues.

    Returns a list of (content, x, y) tuples for each text element whose
    estimated bbox center sits outside every supplied bound. Empty list = clean.

    Note: only flags text that's outside ALL supplied bounds. If you want to
    enforce "every text is inside SOME card", supply all card bounds. If you
    pass an empty parent_bounds list, every text element is flagged.
    """
    elems = _extract_text_elements(svg_xml)
    out: list[tuple] = []
    for e in elems:
        # Use the text's anchor point (x, y) — most reliable; tspan-internal
        # positioning is opaque.
        cx, cy = e["x"], e["y"]
        # Apply text-anchor adjustment to the anchor point for the "is this
        # inside a card" check — for end-anchored text, anchor x is the
        # right edge; bias inward.
        if e["text_anchor"] == "end":
            cx = e["x"] - 2
        elif e["text_anchor"] == "middle":
            cx = e["x"]  # already the center
        inside_any = False
        for (rx, ry, rw, rh) in parent_bounds:
            if (rx - tolerance) <= cx <= (rx + rw + tolerance) and \
               (ry - tolerance) <= cy <= (ry + rh + tolerance):
                inside_any = True
                break
        if not inside_any:
            out.append((e["content"][:60], round(cx, 1), round(cy, 1)))
    return out


def collision_guard(svg_xml: str, *, parent_bounds: list = None,
                    min_gap: int = 4,
                    raise_on_fail: bool = True) -> tuple[list, list]:
    """Convenience: run both collision checks; optionally raise.

    Returns (collisions, out_of_bounds) — both empty lists on success.
    Builders typically call this after build_svg() but before file write:

        svg = build_svg()
        collisions, out_of_bounds = collision_guard(svg, parent_bounds=[...])
        out_path.write_text(svg)
    """
    collisions = check_no_text_collisions(svg_xml, min_gap=min_gap)
    out_of_bounds = check_text_within_bounds(svg_xml, parent_bounds=parent_bounds or [])
    if raise_on_fail and (collisions or out_of_bounds):
        msg_parts = []
        if collisions:
            msg_parts.append(f"{len(collisions)} suspected text collision(s):")
            for a, b, ox, yo in collisions[:10]:
                msg_parts.append(f"  · {a!r}  ↔  {b!r}  (overlap {ox}px x {yo}px)")
        if out_of_bounds:
            msg_parts.append(f"{len(out_of_bounds)} text element(s) outside supplied bounds:")
            for c, cx, cy in out_of_bounds[:10]:
                msg_parts.append(f"  · {c!r}  at ({cx}, {cy})")
        raise ValueError("\n".join(msg_parts))
    return collisions, out_of_bounds
