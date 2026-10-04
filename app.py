"""Well test viewer: merges the gauge file with the water meter file and plots them.

Run locally:   streamlit run app.py
"""
import io
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

NUM = r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
TOTAL = "Water Rate Total"


# ---------------------------------------------------------------- reading
def decode_bytes(data: bytes) -> str:
    """Handles UTF-16 exports (with or without a BOM), UTF-8 and Latin-1."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    if len(data) >= 2 and data[1:2] == b"\x00":
        return data.decode("utf-16-le")
    if len(data) >= 2 and data[0:1] == b"\x00":
        return data.decode("utf-16-be")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def read_table(name: str, data: bytes, skiprows: int) -> pd.DataFrame:
    """Read csv/txt/xlsx. The first row after skiprows becomes the header."""
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data), skiprows=skiprows)
    text = decode_bytes(data)
    lines = text.splitlines()
    if len(lines) <= skiprows + 1:
        raise ValueError("The file has fewer rows than expected. Check the header rows setting.")
    sep = max(["\t", ",", ";"], key=lines[skiprows].count)
    ncols = max(l.count(sep) for l in lines[skiprows:skiprows + 50]) + 2  # room for trailing delimiters
    raw = pd.read_csv(io.StringIO(text), sep=sep, skiprows=skiprows,
                      header=None, names=range(ncols), dtype=str)
    df = raw.iloc[1:].reset_index(drop=True)
    df.columns = raw.iloc[0]
    return df.loc[:, df.columns.notna()]


def clean_unit(u) -> str:
    return str(u).replace("°", "deg").replace(" ", "")


def split_header(col):
    """'ZONE 1 ANNULUS TEMP (deg C)' -> ('ZONE 1 ANNULUS TEMP', 'degC')"""
    m = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", str(col).strip())
    if m:
        return m.group(1).strip(), clean_unit(m.group(2))
    return str(col).strip(), ""


def split_values(s: pd.Series):
    """'2194.8  psia' -> 2194.8, plus the most common unit text found in the cells."""
    s = s.astype(str)
    num = pd.to_numeric(s.str.extract(NUM)[0], errors="coerce")
    rest = s.str.replace(NUM, "", regex=True).str.strip().dropna()
    rest = rest[rest.ne("") & rest.str.lower().ne("nan")]
    unit = clean_unit(rest.mode().iloc[0]) if len(rest) else ""
    return num, unit


@st.cache_data(show_spinner="Reading gauge file...")
def load_main(name, data, title_rows, dayfirst):
    raw = read_table(name, data, title_rows)
    time_col = raw.columns[0]
    cols = {"DateTime": pd.to_datetime(raw[time_col], dayfirst=dayfirst, errors="coerce")}
    units = {"DateTime": ""}
    for col in raw.columns[1:]:
        cname, header_unit = split_header(col)
        cols[cname], cell_unit = split_values(raw[col])
        units[cname] = header_unit or cell_unit
    df = pd.DataFrame(cols)
    if df["DateTime"].notna().sum() == 0:
        raise ValueError(f"No readable dates in the first column. First values: {raw[time_col].head(5).tolist()}")
    df = df.dropna(subset=["DateTime"]).sort_values("DateTime").drop_duplicates("DateTime")
    return df.reset_index(drop=True), units


def find_time_col(names):
    """The date-time column, found by name (Reading, DateTime, Time...) and otherwise the first column."""
    for c in names:
        if re.fullmatch(r"\s*(reading|date\s*time|datetime|date|time|timestamp)\s*", c, re.I):
            return c
    return names[0]


@st.cache_data(show_spinner="Reading water file...")
def load_water(name, data, title_rows, dayfirst):
    w = read_table(name, data, title_rows)
    names = [str(c) for c in w.columns]
    w_units = pd.Series(w.iloc[0].values, index=names)      # units row (bbl/day)
    w = w.iloc[1:].reset_index(drop=True)
    w.columns = names

    time_col = find_time_col(names)
    others = [c for c in names if c != time_col]
    meter_cols = [c for c in others if re.search(r"water\s*rate", c, re.I)] or others[-2:]
    if not meter_cols:
        raise ValueError(f"Could not find water meter columns in: {names}")

    cols = {"Datetime": pd.to_datetime(w[time_col], dayfirst=dayfirst, errors="coerce")}
    units = {}
    for c in meter_cols:
        cols[c], cell_unit = split_values(w[c])
        raw_unit = w_units[c]
        units[c] = clean_unit(raw_unit) if pd.notna(raw_unit) else cell_unit
    df = pd.DataFrame(cols)
    if df["Datetime"].notna().sum() == 0:
        raise ValueError(f"No readable dates in column '{time_col}'. First values: {w[time_col].head(5).tolist()}")
    # One meter runs at a time: a blank counts as zero unless both are blank
    df[TOTAL] = df[meter_cols].sum(axis=1, min_count=1)
    units[TOTAL] = next((u for u in units.values() if u), "")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime")
    return df.reset_index(drop=True), units, meter_cols


# ---------------------------------------------------------------- merging
@st.cache_data(show_spinner="Merging...")
def merge_data(main, water, meter_cols, start, end, how, round_to, per_minute):
    def keep_range(df, col):
        if start is not None:
            df = df[df[col] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df[col] < pd.Timestamp(end) + pd.Timedelta(days=1)]
        return df.copy()

    m = keep_range(main, "DateTime")
    x = keep_range(water, "Datetime")
    water_cols = list(meter_cols) + [TOTAL]

    if round_to:
        m["key"] = m["DateTime"].dt.floor(round_to)
        x["key"] = x["Datetime"].dt.floor(round_to)
    else:
        m["key"], x["key"] = m["DateTime"], x["Datetime"]

    dups = int(x["key"].duplicated().sum())
    x = x.drop(columns="Datetime")
    if per_minute == "mean":
        x = x.groupby("key", as_index=False)[water_cols].mean()
    else:
        x = x.drop_duplicates("key", keep=per_minute)
    m = m.drop_duplicates("key", keep="last")

    merged = m.merge(x, on="key", how=how).sort_values("key")
    merged["DateTime"] = merged["DateTime"].fillna(merged["key"])
    merged = merged.drop(columns="key")
    order = ["DateTime"] + [c for c in main.columns if c != "DateTime"] + water_cols
    return merged[order].reset_index(drop=True), dups


@st.cache_data
def filter_main(main, start, end):
    df = main[(main["DateTime"] >= pd.Timestamp(start)) & (main["DateTime"] < pd.Timestamp(end) + pd.Timedelta(days=1))]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- export
@st.cache_data(show_spinner="Building Excel file...")
def make_xlsx(merged, units) -> bytes:
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell

    cols = list(merged.columns)
    wb = Workbook(write_only=True)
    ws = wb.create_sheet("Merged")
    ws.append(cols)
    ws.append([units.get(c, "") for c in cols])
    body = merged.astype(object).where(merged.notna(), None)
    times = merged["DateTime"].astype(object).tolist()
    for i, row in enumerate(body.itertuples(index=False)):
        t = WriteOnlyCell(ws, value=times[i])
        t.number_format = "mm/dd/yyyy hh:mm:ss"
        ws.append([t] + list(row)[1:])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


@st.cache_data
def make_csv(merged, units) -> bytes:
    cols = list(merged.columns)
    out = merged.copy()
    out["DateTime"] = out["DateTime"].dt.strftime("%m/%d/%Y %H:%M:%S")
    buf = io.StringIO()
    pd.DataFrame([[units.get(c, "") for c in cols]], columns=cols).to_csv(buf, index=False)
    out.to_csv(buf, index=False, header=False)
    return buf.getvalue().encode("utf-8-sig")


# ---------------------------------------------------------------- plot
PALETTE = ["#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#ff7f0e", "#8c564b",
           "#17becf", "#e377c2", "#bcbd22", "#7f7f7f"]
AXIS_SPACING = 0.075          # paper-width fraction between stacked axes on the same side


def pretty(col):
    return col.title() if col.isupper() else col


def auto_title(cols, units, i):
    """A readable axis title from the curves on it."""
    found = {units.get(c, "") for c in cols}
    unit = found.pop() if len(found) == 1 else ""
    stripped = {re.sub(r"^ZONE\s*\d+\s*", "", c, flags=re.I) for c in cols}
    base = pretty(stripped.pop()) if len(stripped) == 1 else (pretty(cols[0]) if len(cols) == 1 else f"Axis {i + 1}")
    return f"{base} ({unit})" if unit else base

def rolling_avg(merged, col, minutes):
    """Time-based moving average of one column, in the same row order as merged."""
    return merged.set_index("DateTime")[col].rolling(f"{minutes}min", min_periods=1).mean()

def build_figure(merged, units, axes, titles, avg_cols, avg_minutes):
    """axes: list of column lists, one per y axis. Axes alternate left, right, left, right..."""
    active = [(i, cols) for i, cols in enumerate(axes) if cols]
    n = len(active)
    n_left, n_right = (n + 1) // 2, n // 2
    x0 = (n_left - 1) * AXIS_SPACING
    x1 = 1 - max(n_right - 1, 0) * AXIS_SPACING

    fig = go.Figure()
    layout = {"xaxis": dict(domain=[x0, x1], title="DateTime", tickformat="%m/%d/%Y\n%H:%M")}
    t = merged["DateTime"]
    color_i = 0
    for k, (i, cols) in enumerate(active):
        ax_id = "y" if k == 0 else f"y{k + 1}"
        axis_color = None
        for col in cols:
            color = PALETTE[color_i % len(PALETTE)]
            color_i += 1
            axis_color = axis_color or color
            has_avg = col in avg_cols
            fig.add_trace(go.Scattergl(
                x=t, y=merged[col], mode="lines", name=pretty(col), yaxis=ax_id,
                line=dict(color=color, width=1 if has_avg else 1.8), opacity=0.35 if has_avg else 1))
            if has_avg:
                avg = merged.set_index("DateTime")[col].rolling(f"{avg_minutes}min", min_periods=1).mean()
                fig.add_trace(go.Scattergl(
                    x=avg.index, y=avg.values, mode="lines", name=f"{pretty(col)} ({avg_minutes} min avg)",
                    yaxis=ax_id, line=dict(color=color, width=2.8)))
        if len(cols) > 1:
            axis_color = "#444444"          # shared axis: neutral colour
        ax = dict(title=dict(text=titles[i].strip() or auto_title(cols, units, i), font=dict(color=axis_color)),
                  tickfont=dict(color=axis_color), showgrid=(k == 0), zeroline=False)
        side_left = (k % 2 == 0)
        stack = k // 2                       # how many axes already sit on this side
        if k > 0:
            ax["overlaying"] = "y"
        ax["side"] = "left" if side_left else "right"
        if stack > 0:
            ax["anchor"] = "free"
            ax["position"] = x0 - stack * AXIS_SPACING if side_left else x1 + stack * AXIS_SPACING
        layout["yaxis" if k == 0 else f"yaxis{k + 1}"] = ax

    fig.update_layout(**layout)
    fig.update_layout(height=640, hovermode="x unified", margin=dict(t=60, r=70, l=70, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


# ---------------------------------------------------------------- app
def main():
    st.set_page_config(page_title="Well test viewer", layout="wide")
    st.title("Well test viewer")

    with st.sidebar:
        st.header("Files")
        main_up = st.file_uploader("ESP Date file From Analysis tab (http://100.80.37.126:3000/)", type=["csv", "txt", "xlsx", "xls"])
        water_up = st.file_uploader("Water meter Data (https://www.wds-solutions.com/?A=sgs-a)", type=["csv", "txt", "xlsx", "xls"])

        with st.expander("File settings"):
            main_title_rows = st.number_input("Rows above the column names (gauge file)", 0, 20, 0)
            water_title_rows = st.number_input("Rows above the column names (water file)", 0, 20, 1)
            main_dayfirst = st.checkbox("Gauge file dates are DD/MM/YYYY", value=True)
            water_dayfirst = st.checkbox("Water file dates are DD/MM/YYYY", value=False)

        with st.expander("Merge settings"):
            how = st.selectbox("Keep", ["outer", "left"], format_func=lambda v: {
                "outer": "All rows from both files", "left": "Only gauge file times"}[v])
            round_to = st.selectbox("Match timestamps to", ["1min", "30s", "5min", "exact"])
            per_minute = st.selectbox("Several water readings in one interval", ["first", "last", "mean"])

    if not main_up:
        st.info("Upload the gauge file in the sidebar to get started. The water meter file is optional.")
        st.stop()

    try:
        main, main_units = load_main(main_up.name, main_up.getvalue(), main_title_rows, main_dayfirst)
        water, water_units, meter_cols = None, {}, []
        if water_up:
            water, water_units, meter_cols = load_water(water_up.name, water_up.getvalue(),
                                                        water_title_rows, water_dayfirst)
    except Exception as e:
        st.error(f"Could not read the files: {e}")
        st.stop()

    if meter_cols:
        st.sidebar.caption("Water meters used: " + ", ".join(meter_cols))
    lo, hi = main["DateTime"].min().date(), main["DateTime"].max().date()
    with st.sidebar:
        st.header("Date range")
        picked = st.date_input("Show data from / to", value=(lo, hi), min_value=lo, max_value=hi)

    if not isinstance(picked, (tuple, list)) or len(picked) != 2:
        st.info("Pick an end date to finish the range.")
        st.stop()
    start, end = picked

    if water is None:
        merged, dups = filter_main(main, start, end), 0
    else:
        merged, dups = merge_data(main, water, meter_cols, start, end, how,
                                  None if round_to == "exact" else round_to, per_minute)
    if merged.empty:
        st.warning("No data in that date range.")
        st.stop()

    units = {**main_units, **water_units}
        # Rename columns and change units before plotting and export
    RENAME = {"RUN TIME SINCE LAST START": "RUN DURATION SINCE LAST START"}
    UNIT_OVERRIDE = {"RUN DURATION SINCE LAST START": "hr"}
    merged = merged.rename(columns=RENAME)
    units = {RENAME.get(c, c): UNIT_OVERRIDE.get(RENAME.get(c, c), u) for c, u in units.items()}

    curves = [c for c in merged.columns if c != "DateTime"]
    
    curves = [c for c in merged.columns if c != "DateTime"]
    default_p = [c for c in curves if re.match(r"ZONE\s*\d+\s*TUBING PRESSURE", c, re.I)] or curves[:1]

    with st.sidebar:
        st.header("Y axes")
        n_axes = int(st.number_input("Number of y axes", 1, 6, 2 if water is not None else 1,
                                     help="Axes alternate left, right, left, right..."))
        axes, titles = [], []
        for i in range(n_axes):
            key = f"axis_{i}"
            if key not in st.session_state:
                st.session_state[key] = default_p if i == 0 else ([TOTAL] if i == 1 and TOTAL in curves else [])
            st.session_state[key] = [c for c in st.session_state[key] if c in curves]   # drop stale picks
            side = "left" if i % 2 == 0 else "right"
            with st.expander(f"Axis {i + 1} ({side})", expanded=i < 2):
                sel = st.multiselect("Curves", curves, key=key,
                                     format_func=lambda c: f"{c} ({units[c]})" if units.get(c) else c)
                ttl = st.text_input("Axis title (optional)", key=f"title_{i}")
            axes.append(sel)
            titles.append(ttl)

        st.header("Moving average")
        picked_curves = [c for a in axes for c in a]
        if "avg_cols" not in st.session_state:
            st.session_state["avg_cols"] = [TOTAL] if TOTAL in picked_curves else []
        st.session_state["avg_cols"] = [c for c in st.session_state["avg_cols"] if c in picked_curves]
        avg_cols = st.multiselect("Add a moving average for", picked_curves, key="avg_cols")
        avg_minutes = int(st.number_input("Averaging window (minutes)", 1, 1440, 15, step=1))
        include_avg = st.checkbox("Include moving averages in downloads", value=True)

    if not picked_curves:
        st.info("Pick at least one curve for an axis.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(merged):,}")
    c2.metric("From", merged["DateTime"].min().strftime("%m/%d/%Y %H:%M"))
    c3.metric("To", merged["DateTime"].max().strftime("%m/%d/%Y %H:%M"))
    if dups and per_minute:
        st.caption(f"{dups:,} water readings shared an interval with another reading. Using the {per_minute} reading.")

    st.plotly_chart(build_figure(merged, units, axes, titles, avg_cols, avg_minutes))

    d1, d2, _ = st.columns([1, 1, 4])
    d1.download_button("Download Excel", make_xlsx(merged, units), "merged_output.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    d2.download_button("Download CSV", make_csv(merged, units), "merged_output.csv", "text/csv")
    export_df, export_units = merged, units
    if include_avg and avg_cols:
        export_df, export_units = merged.copy(), dict(units)
        for col in avg_cols:
            name = f"{col} ({avg_minutes} min avg)"
            export_df[name] = rolling_avg(merged, col, avg_minutes).values
            export_units[name] = units.get(col, "")
            
    with st.expander("Preview merged data"):
        st.dataframe(merged.head(500))


if __name__ == "__main__":
    main()