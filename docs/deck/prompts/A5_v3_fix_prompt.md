# A5 v3 — Final Equation Rendering Fix (matplotlib mathtext → PNG embed)

**Owner**: Cowork (execution)
**Estimated time**: 30-45 min
**Input commit**: `0bfdc72`
**Input files**:
- `docs/deck/assets/diagrams/A5_causal_architecture.svg`
- `docs/deck/assets/diagrams/_build_a5.py`

---

## Context

A5 v2 shipped at `0bfdc72`. Ash visual review:
- ✅ **GRN visualization** — outstanding work, do not touch
- ✅ **Status pill** — preserved correctly
- ✅ **Subtitle trim** — clean
- ❌ **Equation rendering** still broken in all 3 instances
- ❌ **Architectural-requirement footer** has text collision (two elements rendering on top of each other)
- ❌ **Bottom comparison line** `(I − W)⁻¹ dₚ` shows the same rect-I corruption

The pattern: every instance of `(I − W)⁻¹` in the SVG is broken because the rect-I fallback approach displaced adjacent characters in the text flow. The architectural-requirement footer collision is a separate bug from v2's text repositioning work.

v3 is the **soft cap** for A5 iteration. We address all 3 issues decisively or step back and reconsider scope.

---

## Strategic Approach — Stop Fighting Cairosvg's Text Rendering

Cairosvg has a real glyph substitution bug for italic Latin I → turnstile `⊢` that **cannot be fixed via font-family changes** (v2 proved this). The rect-I fallback was a creative attempt but breaks adjacent characters in text flow.

**The correct engineering response is to stop rendering the equation as SVG text entirely.** Render the equation via matplotlib's mathtext (which produces correct math typography), export as PNG, embed the PNG into the SVG via `<image>` tag at the right position.

This is **standard practice** for embedding math in SVG when the renderer has math-rendering limitations.

---

## The Fix — Equation As Embedded PNG (Three Instances)

### Where the equation appears

A5 has three instances of `(I − W)⁻¹` expressions that all need this fix:

1. **Hero equation** (top zone, 56pt equivalent): `ŷ = (I − W)⁻¹ · dₚ`
2. **Component definitions row** (24pt equivalent, third column): `(I − W)⁻¹` with annotation "closed-form propagation"
3. **Bottom comparison line** (12pt equivalent): `Stage 3c separates: dₚ (direct) + (I − W)⁻¹ dₚ (propagated)`

For each instance, render the math expression as a PNG and embed it.

### Implementation Pattern

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io

def render_math_to_base64_png(latex_expr: str, fontsize: int = 56, 
                              color: str = "white", dpi: int = 300) -> tuple:
    """Render a LaTeX-style math expression to a base64-encoded PNG.
    
    Returns (base64_data_uri, width_pixels, height_pixels).
    Use matplotlib mathtext for cross-platform consistency without LaTeX install.
    """
    fig = plt.figure(figsize=(0.01, 0.01))  # tiny figure, will be auto-sized
    fig.patch.set_alpha(0)  # transparent background
    
    # Render math as text on the figure
    text = fig.text(0, 0, latex_expr, fontsize=fontsize, color=color)
    
    # Measure text bbox
    fig.canvas.draw()
    bbox = text.get_window_extent()
    width_px = int(bbox.width * dpi / 72)
    height_px = int(bbox.height * dpi / 72)
    
    # Resize figure to fit text exactly
    fig.set_size_inches(width_px / dpi, height_px / dpi)
    text.set_position((0, 0))  # bottom-left anchor
    
    # Save to in-memory buffer as PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, transparent=True, 
                bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    
    # Encode to base64 data URI
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('ascii')
    data_uri = f"data:image/png;base64,{b64}"
    
    return data_uri, width_px, height_px


def embed_math_in_svg(latex_expr: str, x: int, y: int, 
                      fontsize: int = 56, color: str = "white") -> str:
    """Return an SVG `<image>` tag with the math expression rendered as PNG."""
    data_uri, w, h = render_math_to_base64_png(latex_expr, fontsize, color)
    return f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{data_uri}" />'
```

### LaTeX Expressions For Each Instance

**Hero equation** (top zone):
```python
# matplotlib mathtext syntax (subset of LaTeX, no preamble needed)
hero_expr = r"$\hat{y} = (I - W)^{-1} \cdot d_p$"
# fontsize ~56 equivalent
```

For **color-coded** rendering (W cyan, dₚ lavender, propagation operator green), matplotlib mathtext supports inline color via `\color{}`. However, simpler approach: render the entire equation in white at high contrast, OR render three separate sub-expressions (each in its own color, positioned adjacently). Recommend **single white equation** for v3 — color coding can come back in Phase 4 if desired. **Correctness over decoration.**

**Component definitions** (third column, ~24pt):
```python
comp3_expr = r"$(I - W)^{-1}$"
# fontsize ~24 equivalent, rendered in green or white
```

**Bottom comparison line** (~12pt, inline in a longer text):
```python
# The comparison line has both regular text and math. Two paths:
# Path A: Render the math fragment only as PNG, position next to SVG text
# Path B: Render the entire comparison line as PNG (single image)
# Recommend Path A — keeps the regular SVG text editable; only math becomes image

# The math fragment in this line:
math_frag = r"$(I - W)^{-1} d_p$"
# fontsize ~12 equivalent
```

### Positioning In SVG

For each embedded math image, position it where the original text was:

```python
# Hero equation — center of top zone, large
hero_x = 580
hero_y = 210
svg_parts.append(embed_math_in_svg(hero_expr, hero_x, hero_y, fontsize=56))

# Component definitions third column
comp3_x = 1080
comp3_y = 280
svg_parts.append(embed_math_in_svg(comp3_expr, comp3_x, comp3_y, fontsize=24, color="#4ADE80"))

# Bottom comparison line — math fragment inserted into regular SVG text flow
# Render the math, position next to the existing text "dₚ (direct) + "
math_x = 920
math_y = 870
svg_parts.append(embed_math_in_svg(math_frag, math_x, math_y, fontsize=12))
```

The exact x/y values need tuning to match the v2 layout positions. Use the existing v2 `_build_a5.py` coordinate constants as starting points.

### Remove All Rect-I And tspan Math Logic

Delete from `_build_a5.py`:
- All `<rect>` elements used as I-fallback (3 sites)
- All `<tspan>` elements rendering `(`, `I`, `−`, `W`, `)`, `⁻¹` as separate spans
- All `font-family` overrides on math characters
- The known-FP filter for rect-split adjacencies (no longer needed since equation is one image)

Result: cleaner build code, no more font-substitution risk, math renders correctly via matplotlib.

---

## Fix 2 — Architectural-Requirement Footer Collision

Separate from the equation fix, the footer "Architectural requirement: ρ(W) < 1 enforced by sparsity L1 — guarantees Neumann-series convergence" has **two text elements rendering on top of each other** at y≈440.

Visual evidence (zoomed):
```
Architectural requirementoy spareihy ity sparsity p[W] g1arantees Neumann-series convergence
```

Two text strings collided into garbled output. Same bug class as B2/D1/D2/F1 v1/A5 v2.

### Fix

Investigate `_build_a5.py` for two `<text>` elements at the same y coordinate covering this region. Likely one is from v1 layout, one was added in v2 layout, and the v1 one wasn't removed cleanly.

Render the footer as ONE single text element:
```
Architectural requirement: ρ(W) < 1 enforced by sparsity L1 — guarantees Neumann-series convergence
```

At y≈440, single line, italic muted grey. No duplicate text elements.

For the `ρ(W) < 1` math inline in the footer, **decide once for v3**: either render the whole footer as plain text (accept that `ρ` and `<` render correctly without math typography) or render the `ρ(W) < 1` fragment as embedded PNG via matplotlib mathtext.

**Recommendation: plain SVG text for this footer.** `ρ` (U+03C1) and `<` (U+003C) render correctly in Inter without math substitution issues. The matplotlib mathtext approach is reserved for `(I − W)⁻¹` only (the proven-broken expression).

---

## Acceptance Criteria For A5 v3

### Equation rendering (Fix 1)

- ✓ Hero equation `ŷ = (I − W)⁻¹ · dₚ` renders correctly with all symbols visible:
  - `ŷ`, `=`, `(`, `I`, `−`, `W`, `)`, `⁻¹` (raised), `·`, `dₚ` (with subscript)
- ✓ Component definitions third column `(I − W)⁻¹` renders correctly
- ✓ Bottom comparison line `(I − W)⁻¹ dₚ` renders correctly
- ✓ No rect-I elements remain in the SVG
- ✓ No `<tspan>` math typography that depends on cairosvg font behavior
- ✓ Math is rendered via embedded `<image>` PNGs from matplotlib mathtext

### Footer collision (Fix 2)

- ✓ "Architectural requirement..." footer renders as single readable line
- ✓ No text collision in the footer region
- ✓ `ρ(W) < 1` displays correctly (plain SVG text)

### No regression (preserve v2 wins)

- ✓ GRN visualization unchanged (8 named genes, gradients, edge weights, legend)
- ✓ Status pill unchanged (STAGE 3c · SPEC-LOCKED · Q1-Q2 2027 · post Phase 1)
- ✓ Subtitle unchanged (125 chars)
- ✓ Direct-effect log-FC head block diagram unchanged (only the inline math fragment swaps to PNG)
- ✓ Causal vs predictive comparison rows present
- ✓ Honesty discipline preserved (no "operational" / "validated" / "in production")
- ✓ Section A palette unchanged
- ✓ Header eyebrow + pagination + source footer preserved

### Helper validation

- ✓ Helper smoke at min_gap=2: 0 blocking
- ✓ Helper smoke at min_gap=0 (strictest): 0 blocking
- ✓ No need for rect-split known-FP filter (rect-I logic removed)

### Visual smoke test

Render the PNG. Zoom in at slide-fill scale to:
1. **Hero equation**: every symbol readable, `I` is Latin capital, `⁻¹` is raised
2. **Component definitions row**: third column `(I − W)⁻¹` readable
3. **Architectural-requirement footer**: single readable line, no overlap
4. **Bottom comparison line**: `(I − W)⁻¹ dₚ` readable inline

---

## Implementation Details

### Matplotlib mathtext caveat

matplotlib's mathtext is a LaTeX subset, not full LaTeX. It supports:
- Basic operators: `+`, `-`, `*`, `/`, `=`, `<`, `>`
- Greek letters: `\rho`, `\alpha`, etc.
- Subscript/superscript: `_{}`, `^{}`
- Hat/bar accents: `\hat{y}`, `\bar{x}`
- Parentheses, brackets, fractions

For A5's equations, mathtext is sufficient — no LaTeX install needed, ships with matplotlib.

### Caching consideration

If matplotlib rendering is slow (it can be ~100-500ms per expression), cache the rendered PNGs to disk in `_build_a5.py` working directory:

```python
import hashlib
def cached_math_png(expr, fontsize, color):
    key = hashlib.sha1(f"{expr}|{fontsize}|{color}".encode()).hexdigest()[:12]
    cache_path = Path(f".math_cache/{key}.png")
    if cache_path.exists():
        return cache_path.read_bytes()
    # ... render via matplotlib, save to cache, return bytes
```

Optional optimization — only needed if build feels slow. For 3 expressions, probably not necessary.

### Font matching

matplotlib's default math font is "DejaVu Sans" which has decent math glyphs. For visual consistency with Inter (used elsewhere in the deck), use:

```python
matplotlib.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern
# Or: 'stix' for STIX fonts
# Or: 'custom' with rcParams for Inter (more complex setup)
```

**Recommendation: `stix` fontset.** Renders math symbols clearly, looks professional, matches the visual register of investor-grade typography.

---

## Deliverable

Single commit:

```bash
git add docs/deck/assets/diagrams/A5_causal_architecture.svg \
        docs/deck/assets/diagrams/A5_causal_architecture_preview.png \
        docs/deck/assets/diagrams/_build_a5.py
git commit -m "fix(deck): A5 v3 - equation as matplotlib mathtext PNG embed"
git push origin main
```

Single-line commit message per zsh history-expansion lesson.

---

## What Ash Will Check

1. **Hero equation**: every symbol readable, `I` is Latin capital, `⁻¹` is raised superscript
2. **Footer**: single readable line, no overlap
3. **Bottom comparison**: math fragment inline reads correctly
4. **GRN visualization**: unchanged (v2 was excellent)
5. **Status pill**: unchanged
6. **Overall slide**: investor-grade architectural-depth slide

If A5 v3 still has issues, we step back per the v2 soft cap discussion. Likely next step would be Option 2 (simplify A5, drop the equation entirely, GRN becomes hero) or Option 3 (defer A5, ship 20-slide v2 deck).

---

## What's Out Of Scope

- Modifying GRN visualization (v2 was excellent)
- Modifying status pill, subtitle, log-FC head block diagram (v2 layout preserved)
- Modifying A5 content spec
- Any other SVG
- pptx assembly (separate prompt after A5 v3 ships)

---

## Risks To Flag

1. **matplotlib import overhead**: matplotlib adds ~100ms+ to build script startup. Acceptable for a one-off SVG build but worth noting.

2. **PNG vector loss**: math expressions are now raster, not vector. At 1920×1080 deck scale with `dpi=300` rendering, this is invisible. At zoomed-in print or future 4K, may slightly soften. Acceptable trade-off for correct rendering.

3. **mathtext font fallback**: if cairosvg renders the embedded PNG correctly (likely — PNG is just bitmap data), the equation will look correct regardless of system fonts. PNG embedding bypasses font-fallback entirely.

4. **PNG inline base64 size**: each math PNG is ~5-20KB base64-encoded. Three expressions = ~30-60KB added to SVG size. Acceptable.

5. **Color coding loss in equation**: v3 starts with white-only equation. If you want W cyan / dₚ lavender / (I−W)⁻¹ green color coding back, that's three separate PNG renders positioned adjacently. Phase 4 polish work. Don't add complexity in v3.

6. **If matplotlib mathtext still has issues**: fallback is to use a different rendering tool (PIL.ImageDraw with custom math font, or sympy.preview, or LaTeX installation). matplotlib is the lightest-weight option that ships with Python; if it fails, we have other paths.

---

## After This Lands

If A5 v3 ships clean:
1. Visual verification on Mac
2. **pptx v3 reassembly** — A5 inserts between A4 and B divider, deck grows to 21 slides
3. Phase 4 polish prompt drafted

If A5 v3 still has issues:
1. Step back per v2 soft cap
2. Decide: simplify A5 (Option 2) or defer A5 (Option 3)
3. Ship pptx without A5 if needed
