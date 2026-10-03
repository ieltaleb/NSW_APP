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


@st.cache_data(show_spinner="Reading water file...")
def load_water(name, data, title_rows, dayfirst):
    w = read_table(name, data, title_rows)
    w_units = w.iloc[0]                       # units row (bbl/day)
    w = w.iloc[1:].reset_index(drop=True)
    meter_cols = [str(c) for c in w.columns[1:3]]
    if len(meter_cols) < 2:
        raise ValueError(f"Expected a time column plus two water meter columns, found: {list(w.columns)}")
    w.columns = [str(c) for c in w.columns]
    cols = {"Datetime": pd.to_datetime(w[w.columns[0]], dayfirst=dayfirst, errors="coerce")}
    units = {}
    for c in meter_cols:
        cols[c], cell_unit = split_values(w[c])
        raw_unit = w_units.iloc[list(w.columns).index(c)]
        units[c] = clean_unit(raw_unit) if pd.notna(raw_unit) else cell_unit
    df = pd.DataFrame(cols)
    if df["Datetime"].notna().sum() == 0:
        raise ValueError(f"No readable dates in the water file. First values: {w[w.columns[0]].head(5).tolist()}")
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
def build_figure(merged, units, pressure_cols, avg_minutes):
    rate = merged.set_index("DateTime")[TOTAL]
    rate_avg = rate.rolling(f"{avg_minutes}min", min_periods=1).mean()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    palette = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#7f7f7f"]
    for i, col in enumerate(pressure_cols):
        fig.add_trace(go.Scattergl(x=merged["DateTime"], y=merged[col], mode="lines", name=col.title(),
                                   line=dict(color=palette[i % len(palette)], width=1.8)),
                      secondary_y=False)
    fig.add_trace(go.Scattergl(x=rate.index, y=rate.values, mode="lines", name="Total Water Rate",
                               line=dict(color="rgba(255,127,14,0.35)", width=1)),
                  secondary_y=True)
    fig.add_trace(go.Scattergl(x=rate_avg.index, y=rate_avg.values, mode="lines",
                               name=f"Total Water Rate ({avg_minutes} min avg)",
                               line=dict(color="#d62728", width=2.5)),
                  secondary_y=True)

    p_unit = next((units[c] for c in pressure_cols if units.get(c)), "")
    fig.update_yaxes(title_text=f"Tubing Pressure ({p_unit})", secondary_y=False)
    fig.update_yaxes(title_text=f"Total Water Rate ({units.get(TOTAL, '')})", secondary_y=True,
                     showgrid=False)
    fig.update_xaxes(title_text="DateTime", tickformat="%m/%d/%Y\n%H:%M")
    fig.update_layout(height=620, hovermode="x unified", margin=dict(t=60, r=20, l=20, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


# ---------------------------------------------------------------- app
def main():
    st.set_page_config(page_title="Well test viewer", layout="wide")
    st.title("Well test viewer")

    with st.sidebar:
        st.header("Files")
        main_up = st.file_uploader("Gauge file (pressures and temperatures)", type=["csv", "txt", "xlsx", "xls"])
        water_up = st.file_uploader("Water meter file", type=["csv", "txt", "xlsx", "xls"])

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

    if not (main_up and water_up):
        st.info("Upload the gauge file and the water meter file in the sidebar to get started.")
        st.stop()

    try:
        main, main_units = load_main(main_up.name, main_up.getvalue(), main_title_rows, main_dayfirst)
        water, water_units, meter_cols = load_water(water_up.name, water_up.getvalue(),
                                                    water_title_rows, water_dayfirst)
    except Exception as e:
        st.error(f"Could not read the files: {e}")
        st.stop()

    lo, hi = main["DateTime"].min().date(), main["DateTime"].max().date()
    with st.sidebar:
        st.header("Date range")
        picked = st.date_input("Show data from / to", value=(lo, hi), min_value=lo, max_value=hi)
        st.header("Averaging")
        avg_minutes = st.number_input("Water rate averaging window (minutes)", 1, 1440, 15, step=1)

    if not isinstance(picked, (tuple, list)) or len(picked) != 2:
        st.info("Pick an end date to finish the range.")
        st.stop()
    start, end = picked

    merged, dups = merge_data(main, water, meter_cols, start, end, how,
                              None if round_to == "exact" else round_to, per_minute)
    if merged.empty:
        st.warning("No data in that date range.")
        st.stop()

    units = {**main_units, **water_units}
    default_p = [c for c in merged.columns if re.match(r"ZONE\s*\d+\s*TUBING PRESSURE", c, re.I)]
    with st.sidebar:
        st.header("Left axis")
        pressure_cols = st.multiselect("Pressure curves", [c for c in merged.columns if units.get(c, "").startswith("psi")],
                                       default=default_p)
    if not pressure_cols:
        st.info("Pick at least one pressure curve.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(merged):,}")
    c2.metric("From", merged["DateTime"].min().strftime("%m/%d/%Y %H:%M"))
    c3.metric("To", merged["DateTime"].max().strftime("%m/%d/%Y %H:%M"))
    if dups and per_minute:
        st.caption(f"{dups:,} water readings shared an interval with another reading. Using the {per_minute} reading.")

    st.plotly_chart(build_figure(merged, units, pressure_cols, int(avg_minutes)))

    d1, d2, _ = st.columns([1, 1, 4])
    d1.download_button("Download Excel", make_xlsx(merged, units), "merged_output.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    d2.download_button("Download CSV", make_csv(merged, units), "merged_output.csv", "text/csv")

    with st.expander("Preview merged data"):
        st.dataframe(merged.head(500))


if __name__ == "__main__":
    main()
