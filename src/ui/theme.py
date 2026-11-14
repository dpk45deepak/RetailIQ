"""Visual system for the RetailIQ dashboard: CSS, HTML components and a Plotly template.

Design notes
------------
* Chrome (hero, buttons, nav highlight) uses a teal -> blue -> indigo gradient.
* Data colours (segments) are a SEPARATE, validated categorical palette (see src/utils/config.py).
* Surfaces use translucent neutrals so every component works in both Streamlit light and dark themes.
* Type: Sora for headings and big numbers, DM Sans for everything else, tabular figures for numbers.
"""
from __future__ import annotations

import html as _html
import inspect
import re

import plotly.graph_objects as go
import streamlit as st

from src.utils.config import SEGMENT_COLORS, SEGMENT_COLORS_DARK, SEGMENT_ORDER

TEAL, BLUE, INDIGO = "#0d9488", "#2563eb", "#4f46e5"
ACCENTS = [TEAL, BLUE, INDIGO]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Sora:wght@600;700&display=swap');
.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea, .stApp td, .stApp th,
.stApp button, .stApp [data-baseweb="select"] { font-family: 'DM Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp h1, .stApp h2, .stApp h3, .rq-hero h1, .rq-kpi .v, .rq-big { font-family: 'Sora', 'DM Sans', system-ui, sans-serif; letter-spacing: -0.02em; }
:root { --rq-line: rgba(127,127,127,.22); --rq-card: rgba(127,127,127,.06); --rq-muted: rgba(127,127,127,.95); }
.block-container { padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* ---- sidebar ---- */
section[data-testid="stSidebar"] { border-right: 1px solid var(--rq-line); }
[data-testid="stSidebarNavLink"], [data-testid="stSidebarNav"] a { border-radius: 10px; margin: 2px 6px; transition: background .15s; }
[data-testid="stSidebarNavLink"][aria-current="page"], [data-testid="stSidebarNav"] a[aria-current="page"] {
  background: linear-gradient(90deg, rgba(13,148,136,.20), rgba(79,70,229,.16)); font-weight: 600; }
.rq-side-foot { font-size: .78rem; color: var(--rq-muted); line-height: 1.5; padding: 10px 6px 0; border-top: 1px solid var(--rq-line); margin-top: 14px; }

/* ---- hero ---- */
.rq-hero { position: relative; overflow: hidden; border-radius: 20px; padding: 26px 30px 24px; margin-bottom: 22px; color: #fff;
  background: radial-gradient(900px 260px at 92% -40%, rgba(255,255,255,.22), transparent 60%),
              linear-gradient(118deg, #0f766e 0%, #2563eb 58%, #4f46e5 100%); box-shadow: 0 10px 30px -14px rgba(37,99,235,.55); }
.rq-hero h1 { color: #fff !important; font-size: 1.85rem; font-weight: 700; margin: 0 0 4px; padding: 0; line-height: 1.15; }
.rq-hero p { margin: 0; color: rgba(255,255,255,.88) !important; font-size: .98rem; max-width: 62ch; }
.rq-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.rq-chip { font-size: .76rem; font-weight: 600; padding: 4px 11px; border-radius: 999px; background: rgba(255,255,255,.17); border: 1px solid rgba(255,255,255,.28); color: #fff; }
.rq-chip.warn { background: rgba(251,191,36,.22); border-color: rgba(253,224,71,.55); }

/* ---- kpi cards ---- */
.rq-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin-bottom: 22px; }
.rq-kpi { position: relative; background: var(--rq-card); border: 1px solid var(--rq-line); border-radius: 16px; padding: 16px 18px 15px; overflow: hidden; }
.rq-kpi::before { content: ""; position: absolute; left: 0; top: 0; right: 0; height: 3px; background: var(--acc, #0d9488); }
.rq-kpi .l { font-size: .8rem; font-weight: 500; color: var(--rq-muted); }
.rq-kpi .v { font-size: 1.72rem; font-weight: 700; margin-top: 4px; font-variant-numeric: tabular-nums; line-height: 1.15; }
.rq-kpi .s { font-size: .78rem; color: var(--rq-muted); margin-top: 3px; }

/* ---- generic cards ---- */
.rq-card { background: var(--rq-card); border: 1px solid var(--rq-line); border-radius: 16px; padding: 18px 20px; }
.rq-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }
.rq-section { margin: 26px 0 8px; }
.rq-section h3 { margin: 0; font-size: 1.12rem; font-weight: 600; padding: 0; }
.rq-section p { margin: 2px 0 0; color: var(--rq-muted); font-size: .88rem; }
.rq-badge { display: inline-flex; align-items: center; gap: 7px; padding: 4px 12px 4px 9px; border-radius: 999px; font-size: .82rem; font-weight: 600; border: 1px solid var(--rq-line); background: var(--rq-card); }
.rq-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

/* ---- insight cards ---- */
.rq-insight { border: 1px solid var(--rq-line); border-radius: 16px; padding: 16px 18px; background: var(--rq-card); display: flex; flex-direction: column; gap: 6px; border-left: 4px solid var(--k, #2563eb); }
.rq-insight .tag { font-size: .74rem; font-weight: 700; color: var(--k, #2563eb); }
.rq-insight h4 { margin: 0; font-size: 1.02rem; font-weight: 600; font-family: 'Sora', sans-serif; letter-spacing: -.01em; }
.rq-insight .f { font-size: .9rem; line-height: 1.5; }
.rq-insight .a { font-size: .86rem; line-height: 1.45; color: var(--rq-muted); border-top: 1px dashed var(--rq-line); padding-top: 7px; margin-top: 2px; }

/* ---- recommendations ---- */
.rq-reco { display: grid; grid-template-columns: 44px 1fr 150px; gap: 14px; align-items: center; border: 1px solid var(--rq-line); border-radius: 14px; padding: 13px 16px; margin-bottom: 10px; background: var(--rq-card); }
.rq-rank { width: 36px; height: 36px; border-radius: 11px; display: grid; place-items: center; font-weight: 700; font-family: 'Sora', sans-serif; color: #fff; background: linear-gradient(135deg, #0d9488, #4f46e5); }
.rq-reco .t { font-weight: 600; font-size: .96rem; }
.rq-reco .c { font-size: .76rem; color: var(--rq-muted); }
.rq-reco .r { font-size: .84rem; margin-top: 3px; }
.rq-meter { height: 8px; border-radius: 99px; background: var(--rq-line); overflow: hidden; display: flex; }
.rq-meter i { display: block; height: 100%; }
.rq-score { text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; font-family: 'Sora', sans-serif; margin-bottom: 6px; }
@media (max-width: 720px) { .rq-reco { grid-template-columns: 36px 1fr; } .rq-reco > div:last-child { grid-column: 1 / -1; } .rq-hero h1 { font-size: 1.45rem; } }

/* ---- widgets ---- */
button[kind="primary"] { background: linear-gradient(120deg, #0d9488, #2563eb 60%, #4f46e5); border: 0; border-radius: 10px; }
[data-testid="stTabs"] button[role="tab"] { font-weight: 600; }
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--rq-line); }
.rq-note { font-size: .84rem; color: var(--rq-muted); }
</style>
"""


def setup() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _compact(s: str) -> str:
    return re.sub(r"\n\s*", "", s.strip())


def html(s: str) -> None:
    st.markdown(_compact(s), unsafe_allow_html=True)


def esc(x) -> str:
    return _html.escape(str(x))


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def seg_colors() -> dict[str, str]:
    return SEGMENT_COLORS_DARK if is_dark() else SEGMENT_COLORS


def seg_color(name: str) -> str:
    return seg_colors().get(name.replace(" II", ""), "#94a3b8")


# ---------------------------------------------------------------------------------- components
def hero(title: str, subtitle: str, chips: list[tuple[str, bool]] | None = None) -> None:
    chip_html = "".join(f'<span class="rq-chip{" warn" if warn else ""}">{esc(t)}</span>' for t, warn in (chips or []))
    html(f'<div class="rq-hero"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p><div class="rq-chips">{chip_html}</div></div>')


def kpis(items: list[dict]) -> None:
    """items: dicts with label, value, optional sub and accent (hex)."""
    cards = "".join(
        f'<div class="rq-kpi" style="--acc:{i.get("accent", ACCENTS[n % 3])}"><div class="l">{esc(i["label"])}</div>'
        f'<div class="v">{esc(i["value"])}</div>' + (f'<div class="s">{esc(i["sub"])}</div>' if i.get("sub") else "") + "</div>"
        for n, i in enumerate(items))
    html(f'<div class="rq-kpis">{cards}</div>')


def section(title: str, caption: str = "") -> None:
    html(f'<div class="rq-section"><h3>{esc(title)}</h3>' + (f"<p>{esc(caption)}</p>" if caption else "") + "</div>")


def badge(segment: str) -> str:
    return f'<span class="rq-badge"><span class="rq-dot" style="background:{seg_color(segment)}"></span>{esc(segment)}</span>'


def insight_card(kind: str, title: str, finding: str, action: str) -> str:
    colour, label = {"risk": ("#eb6834", "Risk"), "opportunity": ("#1baf7a", "Opportunity")}.get(kind, ("#2563eb", "Insight"))
    return (f'<div class="rq-insight" style="--k:{colour}"><div class="tag">{label}</div><h4>{esc(title)}</h4>'
            f'<div class="f">{esc(finding)}</div><div class="a"><b>Next step:</b> {esc(action)}</div></div>')


def gbp(x: float, compact: bool = True) -> str:
    if compact and abs(x) >= 1e6:
        return f"£{x / 1e6:,.2f}M"
    if compact and abs(x) >= 1e4:
        return f"£{x / 1e3:,.1f}k"
    return f"£{x:,.0f}"


def meta_chips(meta: dict) -> list[tuple[str, bool]]:
    src = "Demo data (synthetic)" if meta.get("is_synthetic") else "UCI Online Retail II"
    return [(src, bool(meta.get("is_synthetic"))), (f"Snapshot {meta['snapshot_date'][:10]}", False), (f"{meta['n_segments']} segments", False)]


# -------------------------------------------------------------------------------------- charts
def style(fig: go.Figure, height: int = 340, legend: bool = True, title: str | None = None) -> go.Figure:
    dark = is_dark()
    ink, grid = ("#cbd5e1", "rgba(148,163,184,.16)") if dark else ("#475569", "rgba(100,116,139,.16)")
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, system-ui, sans-serif", size=13, color=ink), margin=dict(l=6, r=8, t=44 if title else 14, b=6),
        title=dict(text=title, x=0, xanchor="left", font=dict(family="Sora, DM Sans, sans-serif", size=15, color=("#f1f5f9" if dark else "#0f172a"))) if title else None,
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text="", font=dict(size=12)),
        hoverlabel=dict(bgcolor="#0f172a" if not dark else "#1e293b", font=dict(color="#f8fafc", family="DM Sans"), bordercolor="rgba(0,0,0,0)"),
        colorway=[BLUE, "#1baf7a", INDIGO, "#eda100", "#eb6834", "#e87ba4"], bargap=0.28)
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=grid, tickfont=dict(size=12), automargin=True)
    fig.update_yaxes(gridcolor=grid, zeroline=False, linecolor="rgba(0,0,0,0)", tickfont=dict(size=12), automargin=True)
    return fig


def show(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False}, key=key)


def show_df(df, **kwargs) -> None:
    if "width" in inspect.signature(st.dataframe).parameters:
        st.dataframe(df, hide_index=True, **kwargs)
    else:  # older Streamlit
        st.dataframe(df, hide_index=True, use_container_width=True, **kwargs)


def ordered_segments(present) -> list[str]:
    present = set(present)
    return [s for s in SEGMENT_ORDER if s in present] + sorted(present - set(SEGMENT_ORDER))
