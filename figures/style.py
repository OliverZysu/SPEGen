"""
Shared figure style and data access for all manuscript figures.

Every figure script in ``figures/`` imports from this module so that fonts,
sizes, colours and panel labelling stay identical across the paper.

Design targets (journal single/double column):
    single column  = 89 mm  = 3.50 in
    1.5 column     = 120 mm = 4.72 in
    panel          = 110 mm = 4.33 in   (standalone LaTeX subfigure)
    double column  = 183 mm = 7.20 in

Typeface is Times New Roman throughout. Combined multi-panel figures are
kept for the Word manuscript; each panel is also written as its own PDF
so the LaTeX layout can place them independently.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# --------------------------------------------------------------------------
# Canvas geometry
# --------------------------------------------------------------------------

MM = 1.0 / 25.4
COL1 = 89 * MM
COL15 = 120 * MM
COL_PANEL = 110 * MM
COL2 = 183 * MM

DPI = 600

# --------------------------------------------------------------------------
# Colour system
#
# One neutral ramp for magnitude, one diverging ramp for signed effects and a
# small categorical set. Colours are muted on purpose: saturated defaults look
# amateurish in print and reproduce badly in greyscale.
# --------------------------------------------------------------------------

INK = "#1a1a1a"
MUTED = "#3a3a3a"        # secondary text; kept dark enough to stay readable in print
FAINT = "#d4d4d4"
GRID = "#ececec"

ACCENT = "#2f5d8f"       # primary blue
ACCENT_2 = "#b4553a"     # contrasting warm tone
ACCENT_3 = "#4a7c59"     # green

# Sequential: white -> deep blue. Perceptually monotone in lightness.
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#ffffff", "#dfe8f1", "#b6cbdf", "#85a8c8", "#5583ac", "#2f5d8f", "#1b3a5c"],
)

# Diverging: warm (negative) -> white -> blue (positive), balanced lightness.
DIV = LinearSegmentedColormap.from_list(
    "div_warm_blue",
    ["#8c3d22", "#c07a55", "#e8d5c6", "#ffffff",
     "#cfdce8", "#7fa3c4", "#2f5d8f"],
)

CATEGORICAL = [
    "#2f5d8f", "#b4553a", "#4a7c59", "#8a6d9e",
    "#c9a227", "#4f8a8b", "#a4543a", "#6b7280",
]


def _register_times() -> None:
    """Make Times New Roman visible to matplotlib even before font-cache rebuild."""
    from matplotlib import font_manager

    roots = (
        "/System/Library/Fonts/Supplemental",
        "/Library/Fonts",
        "/usr/share/fonts/truetype/msttcorefonts",
        "/usr/share/fonts/truetype/liberation",
    )
    names = (
        "Times New Roman.ttf", "Times New Roman Bold.ttf",
        "Times New Roman Italic.ttf", "Times New Roman Bold Italic.ttf",
        "Times_New_Roman.ttf", "Times_New_Roman_Bold.ttf",
        "Times_New_Roman_Italic.ttf", "Times_New_Roman_Bold_Italic.ttf",
        "LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf",
        "LiberationSerif-Italic.ttf", "LiberationSerif-BoldItalic.ttf",
    )
    for root in roots:
        for name in names:
            path = os.path.join(root, name)
            if os.path.isfile(path):
                try:
                    font_manager.fontManager.addfont(path)
                except (OSError, RuntimeError, ValueError):
                    pass


def apply_style() -> None:
    """Install the manuscript rcParams. Call once at the top of each script."""
    _register_times()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.titlesize": 9,

        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,

        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "lines.linewidth": 1.1,
        "lines.markersize": 3.0,
        "patch.linewidth": 0.5,

        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,

        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax, letter: str, dx: float = -0.085, dy: float = 1.045) -> None:
    """Bold lower-case panel letter placed in axes coordinates."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def column_label(fig, ax, letter: str, x: float, dy: float = 1.045) -> None:
    """Panel letter at a fixed figure-fraction ``x``, vertically keyed to ``ax``.

    Axes-relative offsets put the letters at different distances from the page
    edge whenever the panels have different widths, which reads as a
    misalignment even though every offset is nominally the same. Pinning the
    horizontal position in figure coordinates lets panels in one column share
    a margin.
    """
    y = ax.get_position().y0 + dy * ax.get_position().height
    return fig.text(x, y, letter, fontsize=9, fontweight="bold", va="bottom",
                    ha="left", color=INK)


def align_labels(fig, groups, pad: float = 0.09) -> None:
    """Move each panel letter to sit ``pad`` inches left of its column.

    A letter placed at a guessed figure fraction is only ever right by
    accident: the content it should hug is the leftmost tick label of the
    column, whose width depends on the strings and the font. Guessing too far
    left opens a blank gutter that makes the whole figure look
    off-centre on the page, which is what this measures away.

    ``groups`` pairs each ``Text`` with the axes that share its margin.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.dpi_scale_trans.inverted()
    width = fig.get_size_inches()[0]
    for text, axes in groups:
        left = min(ax.get_tightbbox(renderer).transformed(inverse).x0
                   for ax in axes)
        text.set_x((left - pad) / width)


def hide_spines(ax, keep: Sequence[str] = ("left", "bottom")) -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(side in keep)


def tight_size(fig) -> Tuple[float, float]:
    """Width and height in inches of the tight bounding box actually exported."""
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    return bbox.width, bbox.height


def fit_width(fig, target: float = COL2, iterations: int = 6,
              tol: float = 0.01) -> None:
    """Shrink the canvas so the exported tight bbox is exactly ``target`` wide.

    Legends, colour bars and rotated tick labels sit outside the axes, so a
    figure declared at 183 mm routinely exports at 200-210 mm. Placing that
    file in a 183 mm column scales every glyph down by the same factor and
    quietly pushes 7 pt labels below the 5 pt floor most journals enforce.

    Axis extents scale with the canvas but text does not, so
    ``tight = span * width + text``; the fixed-point iteration below converges
    in two or three passes.
    """
    for _ in range(iterations):
        current, _ = tight_size(fig)
        delta = target - current
        if abs(delta) < tol:
            return
        w, h = fig.get_size_inches()
        fig.set_size_inches(max(w + delta, 1.0), h, forward=True)


def save(fig, stem: str, out_dir: str, target_width: Optional[float] = COL2) -> str:
    """Write PNG (for Word) and PDF (vector). Returns the PNG path."""
    os.makedirs(out_dir, exist_ok=True)
    if target_width is not None:
        fit_width(fig, target_width)
    png = os.path.join(out_dir, f"{stem}.png")
    pdf = os.path.join(out_dir, f"{stem}.pdf")
    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    w, h = tight_size(fig)
    plt.close(fig)
    print(f"[saved] {pdf}   {w:.2f} x {h:.2f} in")
    return png


def new_panel(width: float, height: float):
    """One-axes figure for a standalone LaTeX panel."""
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def colorbar(fig, im, ax, label: str, ticks: Optional[Sequence[float]] = None):
    """Attach a compact colour bar to ``ax`` and return it."""
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=plt.rcParams["axes.labelsize"], labelpad=3)
    cb.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"],
                      length=1.6, width=0.5, pad=1.5)
    cb.outline.set_linewidth(0.4)
    if ticks is not None:
        cb.set_ticks(list(ticks))
    return cb


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------

# Display names for the 21 descriptors. The raw column names carry a
# ``_value`` suffix that is noise in a figure axis.
DESCRIPTOR_LABELS: Dict[str, str] = {
    "Complexity_value": "Complexity",
    "CovalentUnitCount_value": "Covalent units",
    "DefinedAtomStereoCount_value": "Def. atom stereo",
    "DefinedBondStereoCount_value": "Def. bond stereo",
    "ExactMass_value": "Exact mass",
    "FormalCharge_value": "Formal charge",
    "HBondAcceptor_value": "HB acceptors",
    "HBondDonor_value": "HB donors",
    "HeavyAtomCount_value": "Heavy atoms",
    "MolecularWeight_value": "Mol. weight",
    "MonoIsotopicWeight_value": "Monoisotopic wt.",
    "RotatableBond_value": "Rotatable bonds",
    "TPSA_value": "TPSA",
    "UndefinedAtomStereoCount_value": "Undef. atom stereo",
    "UndefinedBondStereoCount_value": "Undef. bond stereo",
    "FCSP3_value": "Fsp3",
    "AtomCount_value": "Atom count",
    "Kappa1_value": "Kappa-1",
    "Kappa2_value": "Kappa-2",
    "Kappa3_value": "Kappa-3",
    "LogP_value": "logP",
}


def short_cartridge(name: str) -> str:
    """``Cartridge_Class_HLB (Hydrophilic-Lipophilic Balance)`` -> ``HLB``."""
    s = name.replace("Cartridge_Class_", "").strip()
    # Prefer the parenthesised abbreviation when the full name is long.
    if "(" in s and ")" in s:
        head = s.split("(")[0].strip()
        inner = s[s.index("(") + 1:s.rindex(")")].strip()
        if len(head) > 12 and len(inner) <= 12:
            return inner
        if head:
            return head
    return s


def short_solvent(name: str, width: int = 22) -> str:
    s = str(name).strip()
    return s if len(s) <= width else s[: width - 1] + "\u2026"


class SPEData:
    """Pollutant-level descriptors and aggregated protocol labels."""

    def __init__(self, root: str = ".") -> None:
        from spe.data import prepare_data
        from utils.parsing import STEP_NAMES

        prepared = prepare_data(
            data_dir=os.path.join(root, "data"),
            cache_path=os.path.join(root, "data/cache/dataset_cache.pkl"),
            seed=42, train_ratio=0.8, val_ratio=0.1,
            require_spe_info=False, unk_solvent_token="__UNK__",
            use_functional_groups=False, fg_transform="binary", fg_min_pos=10,
            decompose_solvent=False,
            feature_transform="zscore",
        )
        art = prepared.artifacts

        self.step_names: List[str] = list(STEP_NAMES)
        self.descriptor_cols: List[str] = list(art.descriptor_cols)
        self.cartridge_cols: List[str] = list(art.cartridge_cols)
        self.solvent_vocab: List[str] = list(art.solvent_vocab)

        self.y_cartridge = np.asarray(prepared.ds.y_cartridge, dtype=np.int8)
        self.y_step = np.asarray(prepared.ds.y_step, dtype=np.int8)
        self.y_solvent = np.asarray(prepared.ds.y_solvent, dtype=np.int8)
        self.has_info = np.asarray(prepared.ds.has_spe_info).astype(bool)
        self.train_idx = prepared.train_idx
        self.test_idx = prepared.test_idx

        # Descriptors on their ORIGINAL scale. ``prepare_data`` standardises
        # ``ds.xs`` in place, so the unstandardised values are read back from
        # the source table. The CID column is stored as a stringified list
        # (e.g. "['10001604']"), hence the digit extraction before matching.
        import pandas as pd
        table = pd.read_csv(os.path.join(root, "data/processed_molecular_data.csv"))
        table["_cid"] = table["CID"].astype(str).str.extract(r"(\d+)", expand=False)
        table = table.drop_duplicates(subset=["_cid"]).set_index("_cid")

        cids = [str(c).strip() for c in prepared.ds.cids]
        self.raw = table.reindex(cids)[self.descriptor_cols].astype(float)
        self.raw.index = np.arange(len(cids))
        self.cids = np.asarray(cids)

        missing = int(self.raw["LogP_value"].isna().sum())
        if missing > 0.5 * len(cids):
            raise RuntimeError(
                f"descriptor join failed: {missing}/{len(cids)} rows unmatched"
            )

    # -- convenience views -------------------------------------------------

    def descriptor(self, col: str) -> np.ndarray:
        return self.raw[col].to_numpy(dtype=float)

    def top_cartridges(self, k: int = 10, mask: Optional[np.ndarray] = None
                       ) -> List[Tuple[int, str, int]]:
        """``k`` most frequent cartridge classes as (index, short name, count)."""
        y = self.y_cartridge if mask is None else self.y_cartridge[mask]
        counts = y.sum(axis=0)
        order = np.argsort(-counts)[:k]
        return [(int(i), short_cartridge(self.cartridge_cols[i]), int(counts[i]))
                for i in order]

    def solvent_any_step(self) -> np.ndarray:
        """(n, S) indicator: solvent used in *any* step of the protocol."""
        return (self.y_solvent.sum(axis=1) > 0).astype(np.int8)

    def top_solvents(self, k: int = 10, step: Optional[int] = None,
                     drop_unknown: bool = True) -> List[Tuple[int, str, int]]:
        y = self.solvent_any_step() if step is None else self.y_solvent[:, step, :]
        counts = y.sum(axis=0)
        order = np.argsort(-counts)
        out: List[Tuple[int, str, int]] = []
        for i in order:
            name = self.solvent_vocab[i]
            if drop_unknown and name.strip().lower() in {"__unk__", "unk", ""}:
                continue
            out.append((int(i), name, int(counts[i])))
            if len(out) >= k:
                break
        return out
