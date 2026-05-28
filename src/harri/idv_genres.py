import pandas as pd
import numpy as np
import duckdb
import narwhals as nw

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import distinctipy

import re
from hereutil import here
from pathlib import Path
from typing import Callable, cast
from tqdm.auto import tqdm

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

con = duckdb.connect(config=dict(parquet_metadata_cache=True, preserve_insertion_order=False, enable_fsst_vectors=True))
con.sql("SET enable_progress_bar=true;")

vd16 = cast(nw.LazyFrame[duckdb.DuckDBPyRelation], None)
vd17 = cast(nw.LazyFrame[duckdb.DuckDBPyRelation], None)
vd18 = cast(nw.LazyFrame[duckdb.DuckDBPyRelation], None)

c = nw.col
l = nw.lit

def to_narwhals(duckdb_table: duckdb.DuckDBPyRelation) -> nw.LazyFrame[duckdb.DuckDBPyRelation]:
    return nw.from_native(duckdb_table)

def read_parquet(table_name: str, *paths: Path) -> nw.LazyFrame[duckdb.DuckDBPyRelation]:
    if len(paths) == 1:
        files_sql = f"'{paths[0]}'"
    else:
        files_sql = "[" + ", ".join(f"'{p}'" for p in paths) + "]"
    con.sql(f"CREATE OR REPLACE VIEW {table_name} AS SELECT * FROM read_parquet({files_sql});")
    return to_narwhals(con.view(table_name))

groups: dict[str, list] = {}

for file in here(f"data/input/pre").glob("*/*.parquet"):
    print(file)
    table_name = re.sub(r"_\d+$", "", file.stem)
    groups.setdefault(table_name, []).append(file)

for table_name, files in (pbar := tqdm(groups.items())):
    pbar.set_description(f"Registering {table_name}")
    globals()[table_name] = read_parquet(table_name, *files)
    print(f"{table_name} = cast(nw.LazyFrame[DuckDBPyRelation], None)")

genre_df = pd.read_csv(here("data/input/post/hackathon_genre_categorization.csv"))
genre_map = genre_df[["german_genre_term", "Level1", "Level2", "Level3"]].copy()
genre_map["german_genre_term"] = genre_map["german_genre_term"].str.strip()
genre_map = genre_map.dropna(subset=["Level1", "Level2", "Level3"])

genre_list_sql = ", ".join(
    f"'{g.replace(chr(39), chr(39)*2)}'"
    for g in genre_map["german_genre_term"].unique()
)

dataset_ranges = {
    "vd16": (1501, 1600),
    "vd17": (1601, 1700),
    "vd18": (1701, 1800),
}

print("Loading data from parquet files...")
records = []
for table_name, (year_min, year_max) in dataset_ranges.items():
    print(f"  {table_name}...")
    years_df = con.sql(f"""
        SELECT DISTINCT record_number, TRY_CAST(regexp_extract(value, '([0-9]{{4}})', 1) AS INTEGER) AS year
        FROM "{table_name}"
        WHERE field_code = '011@' AND subfield_code = 'a'
          AND TRY_CAST(regexp_extract(value, '([0-9]{{4}})', 1) AS INTEGER) BETWEEN {year_min} AND {year_max}
    """).df()

    genres_df = con.sql(f"""
        SELECT record_number, TRIM(value) AS german_genre_term
        FROM "{table_name}"
        WHERE field_code = '044S' AND subfield_code = 'a'
          AND TRIM(value) IN ({genre_list_sql})
    """).df()

    merged = years_df.merge(genres_df, on="record_number")
    merged["dataset"] = table_name
    records.append(merged)

full_df = pd.concat(records, ignore_index=True)
full_df = full_df.merge(genre_map, on="german_genre_term", how="left")
full_df = full_df.dropna(subset=["Level1", "Level2", "Level3"])
print(f"  Done — {len(full_df):,} genre-record pairs loaded")

# ── City data ────────────────────────────────────────────────────────────────
# Canonical display names for the top-20 cities.
TOP20_CITIES = [
    "Frankfurt, Main",
    "Leipzig",
    "Augsburg",
    "Nürnberg",
    "Köln",
    "Wittenberg",
    "Jena",
    "Straßburg",
    "Berlin",
    "Halle, Saale",
    "Helmstedt",
    "Hamburg",
    "Dresden",
    "Rostock",
    "Wien",
    "Tübingen",
    "Erfurt",
    "Basel",
    "Göttingen",
    "München",
]

def clean_and_classify(city_name):
    if not city_name:
        return None
    c = str(city_name).lower().strip()
    if "leipzig" in c:
        return "Leipzig"
    elif "frankfurt" in c:
        if "oder" in c:
            return None
        else:
            return "Frankfurt, Main"
    elif "jena" in c:
        return "Jena"
    elif "wittenberg" in c:
        return "Wittenberg"
    elif "berlin" in c:
        return "Berlin"
    elif "nürnberg" in c or "nuremberg" in c:
        return "Nürnberg"
    elif "halle" in c:
        return "Halle, Saale"
    elif "hamburg" in c:
        return "Hamburg"
    elif "dresden" in c:
        return "Dresden"
    elif "helmstedt" in c:
        return "Helmstedt"
    elif "rostock" in c:
        return "Rostock"
    elif "wien" in c or "vienna" in c:
        return "Wien"
    elif "straßburg" in c or "strasbourg" in c:
        return "Straßburg"
    elif "augsburg" in c:
        return "Augsburg"
    elif "göttingen" in c or "goettingen" in c:
        return "Göttingen"
    elif "tübingen" in c or "tuebingen" in c:
        return "Tübingen"
    elif "erfurt" in c:
        return "Erfurt"
    elif "köln" in c or "cologne" in c or "colonia" in c:
        return "Köln"
    elif "altdorf" in c:
        return None
    elif "münchen" in c or "munich" in c:
        return "München"
    elif "basel" in c or "basle" in c:
        return "Basel"
    else:
        return None

print("Loading city data from parquet files...")
city_records = []
for table_name in dataset_ranges:
    city_df = con.sql(f"""
        SELECT DISTINCT record_number, TRIM(value) AS raw_city
        FROM "{table_name}"
        WHERE field_code = '033D' AND subfield_code = 'p'
    """).df()
    city_records.append(city_df)

city_df_all = pd.concat(city_records, ignore_index=True)

# Normalise to canonical names, drop anything not in our top-20
city_df_all["city"] = city_df_all["raw_city"].apply(clean_and_classify)
city_df_all = city_df_all.dropna(subset=["city"])
city_df_all = city_df_all[["record_number", "city"]].drop_duplicates()

# Join city onto full_df (left join — records without a top-20 city get NaN)
full_df = full_df.merge(city_df_all, on="record_number", how="left")
print(f"  Done — city column added ({city_df_all['city'].nunique()} distinct cities)")

# ═══════════════════════════════════════════════════════════════════════════
# 2. COLOR MAPS  (consistent colors per category across interactions)
# ═══════════════════════════════════════════════════════════════════════════

def build_color_map(categories):
    cats = sorted(categories)
    colors = distinctipy.get_colors(len(cats))
    return {
        cat: f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        for cat, (r, g, b) in zip(cats, colors)
    }

color_maps = {
    1: build_color_map(full_df["Level1"].dropna().unique()),
    2: build_color_map(full_df["Level2"].dropna().unique()),
    3: build_color_map(full_df["Level3"].dropna().unique()),
}

# ── Parent-child lookup maps ─────────────────────────────────────────────────
l1_to_l2 = (full_df.drop_duplicates(["Level1","Level2"])
             .groupby("Level1")["Level2"].apply(list).to_dict())
l2_to_l3 = (full_df.drop_duplicates(["Level2","Level3"])
             .groupby("Level2")["Level3"].apply(list).to_dict())

all_l1 = sorted(full_df["Level1"].dropna().unique())
all_l2 = sorted(full_df["Level2"].dropna().unique())
all_l3 = sorted(full_df["Level3"].dropna().unique())

def make_opts(all_items, checked, search=""):
    """Checked items always shown first, then search-filtered unchecked items."""
    checked = set(checked or [])
    search  = (search or "").lower()
    checked_opts   = [{"label": x, "value": x} for x in all_items if x in checked]
    unchecked_opts = [{"label": x, "value": x} for x in all_items
                      if x not in checked and (not search or search in x.lower())]
    return checked_opts + unchecked_opts

# ── Styles ───────────────────────────────────────────────────────────────────
SIDEBAR = {
    "width": "300px", "minWidth": "300px", "padding": "24px 20px",
    "background": "#f8f9fa", "borderRight": "1px solid #dee2e6",
    "overflowY": "auto", "height": "100vh", "boxSizing": "border-box",
}
MAIN = {
    "flex": "1", "padding": "20px", "height": "100vh",
    "boxSizing": "border-box", "position": "relative", "overflow": "hidden",
}
PANEL_BASE = {
    "position": "absolute", "top": "50px", "right": "0",
    "height": "88%", "background": "white",
    "border": "1px solid #ced4da", "borderRadius": "8px 0 0 8px",
    "boxShadow": "-4px 0 16px rgba(0,0,0,0.12)",
    "zIndex": "100", "display": "flex", "flexDirection": "row", "overflow": "hidden",
}
COL_BASE = {
    "width": "220px", "minWidth": "220px", "padding": "14px",
    "display": "flex", "flexDirection": "column",
    "height": "100%", "boxSizing": "border-box",
    "borderRight": "1px solid #e9ecef",
}
LABEL = {
    "fontWeight": "600", "fontSize": "11px", "textTransform": "uppercase",
    "letterSpacing": "0.8px", "color": "#6c757d",
    "marginTop": "20px", "marginBottom": "8px", "display": "block",
}
SEARCH_INPUT = {
    "width": "100%", "padding": "5px 8px", "border": "1px solid #ced4da",
    "borderRadius": "4px", "fontSize": "12px",
    "marginBottom": "6px", "boxSizing": "border-box",
}
CLEAR_BTN = {
    "fontSize": "11px", "padding": "2px 10px", "marginBottom": "8px",
    "cursor": "pointer", "border": "1px solid #ced4da",
    "borderRadius": "4px", "background": "#f8f9fa", "color": "#495057",
}

def filter_column(level_num, title, all_items, hidden=False):
    style = {**COL_BASE, "display": "none" if hidden else "flex"}
    if level_num == 3:
        style["borderRight"] = "none"
    return html.Div(id=f"l{level_num}-col", style=style, children=[
        html.Div(title, style={
            "fontWeight": "700", "fontSize": "13px", "color": "#212529",
            "paddingBottom": "8px", "marginBottom": "10px",
            "borderBottom": "1px solid #e9ecef",
        }),
        dcc.Input(id=f"l{level_num}-search", placeholder="Search…",
                  debounce=True, style=SEARCH_INPUT),
        html.Button("Clear", id=f"l{level_num}-clear",
                    n_clicks=0, style=CLEAR_BTN),
        html.Div(
            dcc.Checklist(
                id=f"l{level_num}-checklist",
                options=[{"label": x, "value": x} for x in all_items],
                value=[],
                labelStyle={"display": "block", "marginBottom": "5px",
                            "fontSize": "12px", "cursor": "pointer"},
                inputStyle={"marginRight": "6px"},
            ),
            style={"overflowY": "auto", "flex": "1", "paddingRight": "4px"},
        ),
    ])

# ── App ──────────────────────────────────────────────────────────────────────
app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(
    style={"display": "flex", "fontFamily": "'Segoe UI', sans-serif", "height": "100vh"},
    children=[
        dcc.Store(id="panel-open", data=False),
        dcc.Store(id="prev-l1",    data=[]),
        dcc.Store(id="prev-l2",    data=[]),

        # ── Sidebar ──────────────────────────────────────────────────────
        html.Div(style=SIDEBAR, children=[
            html.H4("Genre Explorer", style={"margin": "0 0 2px 0"}),
            html.Small("VD16 · VD17 · VD18 (1501–1800)",
                       style={"color": "#868e96"}),
            html.Hr(),
            html.Span("Timeline", style=LABEL),
            dcc.RangeSlider(
                id="year-slider", min=1501, max=1800, step=1,
                value=[1501, 1800],
                marks={
                    1501: "1501",
                    1600: {"label": "1600", "style": {"color": "#c0392b",
                                                      "fontWeight": "bold"}},
                    1700: {"label": "1700", "style": {"color": "#c0392b",
                                                      "fontWeight": "bold"}},
                    1800: "1800",
                },
                tooltip={"placement": "bottom", "always_visible": True},
                allowCross=False,
            ),
            html.Hr(),
            html.Span("Display Level", style=LABEL),
            dcc.RadioItems(
                id="display-level",
                options=[
                    {"label": " Level 0 — Total",     "value": 0},
                    {"label": " Level 1 — General",   "value": 1},
                    {"label": " Level 2 — Mid-level", "value": 2},
                    {"label": " Level 3 — Specific",  "value": 3},
                ],
                value=1,
                labelStyle={"display": "block", "marginBottom": "6px",
                            "fontSize": "13px"},
            ),
            html.Hr(),
            html.Span("Smoothing (years)", style=LABEL),
            dcc.Slider(
                id="smoothing-slider", min=1, max=25, step=1, value=5,
                marks={1:"1", 5:"5", 10:"10", 15:"15", 20:"20", 25:"25"},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Hr(),
            html.Span("Publishing City", style=LABEL),
            html.Div([
                html.Button("Select all",   id="city-select-all",   n_clicks=0,
                            style={**CLEAR_BTN, "marginRight": "4px"}),
                html.Button("Deselect all", id="city-deselect-all", n_clicks=0,
                            style=CLEAR_BTN),
            ], style={"marginBottom": "6px"}),
            dcc.Dropdown(
                id="city-dropdown",
                options=[{"label": c, "value": c} for c in TOP20_CITIES],
                value=[],
                multi=True,
                placeholder="No filter — showing all cities…",
                style={"fontSize": "12px"},
            ),
            html.Div(
                id="city-mode-container",
                style={"display": "none"},
                children=[
                    html.Span("City display mode", style={**LABEL, "marginTop": "12px"}),
                    dcc.RadioItems(
                        id="city-mode",
                        options=[
                            {"label": " Combined", "value": "combined"},
                            {"label": " Separate", "value": "separate"},
                        ],
                        value="combined",
                        labelStyle={"display": "block", "marginBottom": "6px",
                                    "fontSize": "13px"},
                    ),
                ],
            ),
        ]),

        # ── Main area ─────────────────────────────────────────────────────
        html.Div(style=MAIN, children=[
            html.Button("⚙ Filter Genres", id="filter-btn", style={
                "marginBottom": "8px", "padding": "6px 16px",
                "background": "#0072B2", "color": "white",
                "border": "none", "borderRadius": "4px",
                "cursor": "pointer", "fontSize": "13px",
            }),
            html.Div(id="filter-panel",
                     style={**PANEL_BASE, "display": "none"},
                     children=[
                         filter_column(1, "Level 1 — General",   all_l1, hidden=False),
                         filter_column(2, "Level 2 — Mid-level", all_l2, hidden=True),
                         filter_column(3, "Level 3 — Specific",  all_l3, hidden=True),
                     ]),
            dcc.Graph(id="main-chart",
                      style={"height": "calc(100% - 44px)"}),
        ]),
    ]
)

# ── Callbacks ────────────────────────────────────────────────────────────────

# 1. Toggle panel open / closed
@app.callback(
    Output("filter-panel", "style"),
    Output("panel-open",   "data"),
    Input("filter-btn",    "n_clicks"),
    State("panel-open",    "data"),
    State("display-level", "value"),
    prevent_initial_call=True,
)
def toggle_panel(_, is_open, display_level):
    if is_open:
        return {**PANEL_BASE, "display": "none"}, False
    n_cols = max(display_level, 1)
    return {**PANEL_BASE, "display": "flex", "width": f"{n_cols * 220}px"}, True


# 2. Resize panel when display level changes (only if already open)
@app.callback(
    Output("filter-panel", "style", allow_duplicate=True),
    Input("display-level", "value"),
    State("panel-open",    "data"),
    prevent_initial_call=True,
)
def resize_panel(display_level, is_open):
    if not is_open:
        return {**PANEL_BASE, "display": "none"}
    n_cols = max(display_level, 1)
    return {**PANEL_BASE, "display": "flex", "width": f"{n_cols * 220}px"}


# 3. Show / hide columns
@app.callback(
    Output("l2-col", "style"),
    Output("l3-col", "style"),
    Input("display-level", "value"),
)
def toggle_cols(display_level):
    l2 = {**COL_BASE, "display": "flex" if display_level >= 2 else "none"}
    l3 = {**COL_BASE, "display": "flex" if display_level >= 3 else "none",
          "borderRight": "none"}
    return l2, l3


# 4-6. Search — options update on both search text change and value change
#      so that newly cascade-checked items always appear at the top
@app.callback(Output("l1-checklist", "options"),
              Input("l1-search",    "value"),
              Input("l1-checklist", "value"))
def search_l1(q, v): return make_opts(all_l1, v, q)

@app.callback(Output("l2-checklist", "options"),
              Input("l2-search",    "value"),
              Input("l2-checklist", "value"))
def search_l2(q, v): return make_opts(all_l2, v, q)

@app.callback(Output("l3-checklist", "options"),
              Input("l3-search",    "value"),
              Input("l3-checklist", "value"))
def search_l3(q, v): return make_opts(all_l3, v, q)


# 7. L1 → L2 cascade
@app.callback(
    Output("l2-checklist", "value"),
    Output("prev-l1",      "data"),
    Input("l1-checklist",  "value"),
    State("prev-l1",       "data"),
    State("l2-checklist",  "value"),
    prevent_initial_call=True,
)
def cascade_l1(l1_now, l1_prev, l2_now):
    l1_now, l1_prev = set(l1_now or []), set(l1_prev or [])
    l2_set = set(l2_now or [])

    for l1 in l1_now - l1_prev:               # newly checked → add children
        l2_set.update(l1_to_l2.get(l1, []))

    still_via_l1 = {c for l1 in l1_now for c in l1_to_l2.get(l1, [])}
    for l1 in l1_prev - l1_now:               # newly unchecked → remove orphaned children
        for child in l1_to_l2.get(l1, []):
            if child not in still_via_l1:
                l2_set.discard(child)

    return list(l2_set), list(l1_now)


# 8. L2 → L3 cascade
@app.callback(
    Output("l3-checklist", "value"),
    Output("prev-l2",      "data"),
    Input("l2-checklist",  "value"),
    State("prev-l2",       "data"),
    State("l3-checklist",  "value"),
    prevent_initial_call=True,
)
def cascade_l2(l2_now, l2_prev, l3_now):
    l2_now, l2_prev = set(l2_now or []), set(l2_prev or [])
    l3_set = set(l3_now or [])

    for l2 in l2_now - l2_prev:
        l3_set.update(l2_to_l3.get(l2, []))

    still_via_l2 = {c for l2 in l2_now for c in l2_to_l3.get(l2, [])}
    for l2 in l2_prev - l2_now:
        for child in l2_to_l3.get(l2, []):
            if child not in still_via_l2:
                l3_set.discard(child)

    return list(l3_set), list(l2_now)


# 9-11. Clear buttons
@app.callback(Output("l1-checklist", "value", allow_duplicate=True),
              Input("l1-clear", "n_clicks"), prevent_initial_call=True)
def clear_l1(_): return []

@app.callback(Output("l2-checklist", "value", allow_duplicate=True),
              Input("l2-clear", "n_clicks"), prevent_initial_call=True)
def clear_l2(_): return []

@app.callback(Output("l3-checklist", "value", allow_duplicate=True),
              Input("l3-clear", "n_clicks"), prevent_initial_call=True)
def clear_l3(_): return []


# 12. Chart
@app.callback(
    Output("main-chart",      "figure"),
    Input("year-slider",      "value"),
    Input("display-level",    "value"),
    Input("l1-checklist",     "value"),
    Input("l2-checklist",     "value"),
    Input("l3-checklist",     "value"),
    Input("smoothing-slider", "value"),
    Input("city-dropdown",    "value"),
    Input("city-mode",        "value"),
)
def update_chart(year_range, display_level, l1_vals, l2_vals, l3_vals, smoothing,
                 city_vals, city_mode):
    year_min, year_max = year_range
    l1_vals   = l1_vals   or []
    l2_vals   = l2_vals   or []
    l3_vals   = l3_vals   or []
    city_vals = city_vals or []

    dff = full_df[(full_df["year"] >= year_min) & (full_df["year"] <= year_max)].copy()
    full_years = pd.RangeIndex(year_min, year_max + 1)

    # ── Genre filter ─────────────────────────────────────────────────────────
    if l1_vals or l2_vals or l3_vals:
        if display_level == 1:
            if l1_vals: dff = dff[dff["Level1"].isin(l1_vals)]
        elif display_level == 2:
            if l2_vals:       dff = dff[dff["Level2"].isin(l2_vals)]
            elif l1_vals:     dff = dff[dff["Level1"].isin(l1_vals)]
        elif display_level == 3:
            if l3_vals:       dff = dff[dff["Level3"].isin(l3_vals)]
            elif l2_vals:     dff = dff[dff["Level2"].isin(l2_vals)]
            elif l1_vals:     dff = dff[dff["Level1"].isin(l1_vals)]

    # ── City filter ──────────────────────────────────────────────────────────
    city_active = bool(city_vals)
    if city_active:
        dff = dff[dff["city"].isin(city_vals)]

    def smooth(s):
        s = s.reindex(full_years, fill_value=0)
        return s.rolling(smoothing, center=True, min_periods=1).mean() if smoothing > 1 else s

    # Line-dash cycle for separate-city mode
    DASH_STYLES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]

    def add_traces(col, cmap, subset=None, dash="solid", name_suffix=""):
        src = subset if subset is not None else dff
        for cat in sorted(src[col].dropna().unique()):
            cat_df = src[src[col] == cat]
            if subset is None or city_mode == "combined":
                cat_df = cat_df.drop_duplicates(subset=["record_number", col])
            y = smooth(cat_df.groupby("year").size())
            label = f"{cat}{name_suffix}"
            fig.add_trace(go.Scatter(
                x=list(full_years), y=y.values, mode="lines", name=label,
                line=dict(width=2, color=cmap.get(cat, "#555"), dash=dash),
                hovertemplate=f"<b>{label}</b><br>Year: %{{x}}<br>Count: %{{y:.1f}}<extra></extra>",
            ))

    fig = go.Figure()
    for b, lbl in [(1600, "VD16/VD17"), (1700, "VD17/VD18")]:
        if year_min < b < year_max:
            fig.add_vline(x=b, line_dash="dot", line_color="#adb5bd", line_width=1.5,
                          annotation_text=lbl, annotation_position="top right",
                          annotation_font=dict(size=10, color="#868e96"))

    # ── No city filter — original behaviour ──────────────────────────────────
    if not city_active:
        if display_level == 0:
            unique_dff = dff.drop_duplicates(subset=["record_number"])
            y = smooth(unique_dff.groupby("year").size())
            fig.add_trace(go.Scatter(
                x=list(full_years), y=y.values, mode="lines", name="All genres",
                line=dict(width=2.5, color="#0072B2"),
                hovertemplate="Year: %{x}<br>Count: %{y:.1f}<extra></extra>",
            ))
        elif display_level == 1: add_traces("Level1", color_maps[1])
        elif display_level == 2: add_traces("Level2", color_maps[2])
        else:                    add_traces("Level3", color_maps[3])

    # ── City filter active — combined ─────────────────────────────────────────
    elif city_mode == "combined":
        if display_level == 0:
            unique_dff = dff.drop_duplicates(subset=["record_number"])
            y = smooth(unique_dff.groupby("year").size())
            cities_label = ", ".join(sorted(city_vals))
            fig.add_trace(go.Scatter(
                x=list(full_years), y=y.values, mode="lines", name=cities_label,
                line=dict(width=2.5, color="#0072B2"),
                hovertemplate="Year: %{x}<br>Count: %{y:.1f}<extra></extra>",
            ))
        elif display_level == 1: add_traces("Level1", color_maps[1])
        elif display_level == 2: add_traces("Level2", color_maps[2])
        else:                    add_traces("Level3", color_maps[3])

    # ── City filter active — separate ─────────────────────────────────────────
    else:
        for i, city in enumerate(sorted(city_vals)):
            dash = DASH_STYLES[i % len(DASH_STYLES)]
            city_subset = dff[dff["city"] == city]
            if display_level == 0:
                y = smooth(city_subset.groupby("year").size())
                fig.add_trace(go.Scatter(
                    x=list(full_years), y=y.values, mode="lines", name=city,
                    line=dict(width=2.5, color="#0072B2", dash=dash),
                    hovertemplate=f"<b>{city}</b><br>Year: %{{x}}<br>Count: %{{y:.1f}}<extra></extra>",
                ))
            elif display_level == 1:
                add_traces("Level1", color_maps[1], subset=city_subset,
                           dash=dash, name_suffix=f" — {city}")
            elif display_level == 2:
                add_traces("Level2", color_maps[2], subset=city_subset,
                           dash=dash, name_suffix=f" — {city}")
            else:
                add_traces("Level3", color_maps[3], subset=city_subset,
                           dash=dash, name_suffix=f" — {city}")

    level_names = {0: "Total", 1: "Level 1", 2: "Level 2", 3: "Level 3"}
    smooth_note = f"  ({smoothing}-yr avg)" if smoothing > 1 else ""
    city_note   = ""
    if city_active:
        city_note = f"  · {', '.join(sorted(city_vals))}" if city_mode == "combined" else "  · cities separate"
    fig.update_layout(
        title=dict(
            text=f"Publications by Genre — {level_names[display_level]}{smooth_note}{city_note}",
            font=dict(size=15)),
        xaxis=dict(
            title="Year", range=[year_min, year_max], tickmode="linear",
            dtick=10 if (year_max - year_min) <= 150 else 25,
            tickangle=45, gridcolor="#e9ecef"),
        yaxis=dict(title="Publications per year", gridcolor="#e9ecef"),
        legend=dict(
            orientation="v", x=1.01, y=1, xanchor="left",
            font=dict(size=11), bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#dee2e6", borderwidth=1),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified", margin=dict(l=60, r=20, t=55, b=70),
    )
    return fig


# 13. Select / deselect all cities
@app.callback(
    Output("city-dropdown", "value"),
    Input("city-select-all",   "n_clicks"),
    Input("city-deselect-all", "n_clicks"),
    prevent_initial_call=True,
)
def city_select_all(n_all, n_none):
    from dash import ctx
    if ctx.triggered_id == "city-select-all":
        return TOP20_CITIES
    return []


# 14. Show / hide city-mode toggle based on whether any city is selected
@app.callback(
    Output("city-mode-container", "style"),
    Input("city-dropdown", "value"),
)
def toggle_city_mode_visibility(city_vals):
    if city_vals:
        return {"display": "block"}
    return {"display": "none"}


app.run(debug=False, jupyter_mode="external")
