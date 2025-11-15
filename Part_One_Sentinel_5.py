"""
flare_tools.py

Utilities for:
- Initializing Google Earth Engine
- Reading point locations from Excel
- Building Sentinel-5P image collections
- Extracting time series at a point or within an area
- Plotting daily time series and monthly boxplots
- Creating labeled map thumbnails
- Building a simple PDF report

You can import this module in Colab after cloning or installing from GitHub.
"""

import os
from collections import defaultdict
from datetime import datetime

import ee
import numpy as np
import pandas as pd
import statistics
import requests
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.interpolate import make_interp_spline

from PIL import Image as PILImage, ImageDraw, ImageFont

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage,
    PageBreak,
)

# ---------------------------------------------------------------------
# Global font config for matplotlib
# ---------------------------------------------------------------------
_available_fonts = {f.name for f in fm.fontManager.ttflist}
if "Times New Roman" in _available_fonts:
    plt.rcParams["font.family"] = "Times New Roman"
else:
    plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 14


# ---------------------------------------------------------------------
# 1. EARTH ENGINE INITIALIZATION
# ---------------------------------------------------------------------
def init_earth_engine(
    project: str = "gee-project-443416",
    use_highvolume: bool = True,
    authenticate: bool = True,
):
    """
    Initialize Earth Engine.

    Parameters
    ----------
    project : str
        GEE project ID.
    use_highvolume : bool
        Use high-volume API URL.
    authenticate : bool
        If True, call ee.Authenticate() first (for Colab).
    """
    if authenticate:
        ee.Authenticate()

    kwargs = {"project": project}
    if use_highvolume:
        kwargs["opt_url"] = "https://earthengine-highvolume.googleapis.com"

    ee.Initialize(**kwargs)


# ---------------------------------------------------------------------
# 2. INPUT DATA: EXCEL + GEOMETRIES
# ---------------------------------------------------------------------
def load_site_from_excel(
    excel_path: str,
    row_index: int = 0,
    name_col: str = "Name",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
):
    """
    Load a single site from an Excel file and return:
    - row (pandas Series)
    - ee.Geometry.Point
    - ee.FeatureCollection of one feature

    Parameters
    ----------
    excel_path : str
        Path to Excel file.
    row_index : int
        Index of row to use (0-based).
    name_col, lat_col, lon_col : str
        Column names in the Excel file.

    Returns
    -------
    row : pandas.Series
    point : ee.Geometry.Point
    fc   : ee.FeatureCollection
    """
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    row = df.iloc[row_index]

    name = str(row[name_col]).strip()
    lat = float(row[lat_col])
    lon = float(row[lon_col])

    point = ee.Geometry.Point([lon, lat])
    feature = ee.Feature(point, {"name": name})
    fc = ee.FeatureCollection([feature])

    return row, point, fc


def make_circular_boundary(point: ee.Geometry, radius_km: float = 50.0) -> ee.Geometry:
    """
    Create a circular boundary (buffered bounds) around a point.

    Parameters
    ----------
    point : ee.Geometry
        Center point.
    radius_km : float
        Radius in kilometers.

    Returns
    -------
    ee.Geometry
        Buffered bounds geometry.
    """
    radius_m = radius_km * 1000.0
    return point.buffer(radius_m).bounds()


# ---------------------------------------------------------------------
# 3. SENTINEL-5P COLLECTIONS
# ---------------------------------------------------------------------
def get_sentinel5p_collections():
    """
    Return a dictionary of Sentinel-5P ImageCollections with the
    main band already selected.

    Returns
    -------
    dict[str, ee.ImageCollection]
    """
    return {
        "UV_Aerosol": ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_AER_AI").select(
            "absorbing_aerosol_index"
        ),
        "CO": ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CO").select(
            "CO_column_number_density"
        ),
        "HCHO": ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_HCHO").select(
            "tropospheric_HCHO_column_number_density"
        ),
        "NO2": ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2").select(
            "tropospheric_NO2_column_number_density"
        ),
        "O3": ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_O3").select(
            "O3_column_number_density"
        ),
        "SO2": ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_SO2").select(
            "SO2_column_number_density"
        ),
        # Add CH4 or CLOUD if needed
        # "CH4": ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CH4").select(
        #     "CH4_column_volume_mixing_ratio_dry_air"
        # ),
    }


# ---------------------------------------------------------------------
# 4. TIME SERIES EXTRACTION
# ---------------------------------------------------------------------
def _time_series_features(collection, band_name, geometry, start_date, end_date):
    """
    Internal helper: returns an ee.FeatureCollection with 'date' and 'value'.
    """
    coll_f = collection.filterDate(start_date, end_date).filterBounds(geometry)

    def _format_feature(img):
        value = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=1000,
            maxPixels=1e13,
        ).get(band_name)
        return ee.Feature(
            None,
            {
                "date": img.date().format("YYYY-MM-dd"),
                "value": value,
            },
        )

    feats = coll_f.map(_format_feature).filter(ee.Filter.notNull(["value"]))
    return feats


def extract_daily_series(
    collection,
    band_name: str,
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Extract mean daily values for a band over a geometry.

    Parameters
    ----------
    collection : ee.ImageCollection
    band_name  : str
    geometry   : ee.Geometry
    start_date : str (YYYY-MM-DD)
    end_date   : str (YYYY-MM-DD)

    Returns
    -------
    pandas.DataFrame with columns ['date', 'value'].
    """
    feats = _time_series_features(collection, band_name, geometry, start_date, end_date)
    feat_list = feats.getInfo()["features"]

    grouped = defaultdict(list)
    for f in feat_list:
        props = f["properties"]
        grouped[props["date"]].append(props["value"])

    dates = []
    values = []
    for date_str, vals in sorted(grouped.items()):
        dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
        values.append(statistics.mean(vals))

    return pd.DataFrame({"date": dates, "value": values})


def extract_all_daily_series(
    collections: dict,
    geometry: ee.Geometry,
    start_date: str,
    end_date: str,
) -> dict:
    """
    Extract daily time series for multiple pollutant collections.

    Parameters
    ----------
    collections : dict
        {pollutant_name: ee.ImageCollection}
    geometry : ee.Geometry
    start_date, end_date : str

    Returns
    -------
    dict[str, pandas.DataFrame]
        Each DataFrame has columns ['date', 'value'].
    """
    result = {}
    for name, coll in collections.items():
        first = coll.first()
        if first is None:
            continue
        band_name = ee.Image(first).bandNames().get(0).getInfo()
        df = extract_daily_series(coll, band_name, geometry, start_date, end_date)
        if not df.empty:
            result[name] = df
    return result


# ---------------------------------------------------------------------
# 5. SMOOTHING + PLOTTING
# ---------------------------------------------------------------------
def smooth_excel_style(dates, values, points: int = 200):
    """
    Smooth line similar to Excel's smoothed line using cubic spline.

    Parameters
    ----------
    dates : sequence[datetime]
    values : sequence[float]
    points : int
        Number of points in the smoothed curve.

    Returns
    -------
    dates_smooth, values_smooth
    """
    if len(values) < 4:
        return dates, values

    x_numeric = np.array([d.timestamp() for d in dates])
    y = np.array(values, dtype=float)

    spline = make_interp_spline(x_numeric, y, k=3)
    x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), points)
    y_smooth = spline(x_smooth)
    dates_smooth = [datetime.fromtimestamp(ts) for ts in x_smooth]
    return dates_smooth, y_smooth


def plot_daily_series(
    series_dict: dict,
    output_path: str = "point_time_series.png",
    dpi: int = 300,
):
    """
    Plot daily time series for multiple pollutants.

    Parameters
    ----------
    series_dict : dict[str, pandas.DataFrame]
        DataFrames with columns ['date', 'value'].
    output_path : str
        PNG output file path.
    """
    n = len(series_dict)
    if n == 0:
        raise ValueError("No time series to plot.")

    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14, 4.5 * nrows),
        dpi=dpi,
    )
    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, (name, df) in enumerate(series_dict.items()):
        ax = axes[i]
        dates = list(df["date"])
        vals = list(df["value"])
        d_smooth, v_smooth = smooth_excel_style(dates, vals)

        ax.plot(d_smooth, v_smooth, linewidth=2)
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{name} (mol/m²)", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    # Hide any unused subplots
    for j in range(len(series_dict), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_boxplots(
    series_dict: dict,
    output_path: str = "monthly_boxplot_pollutants.png",
    dpi: int = 300,
):
    """
    Create monthly boxplots (Jan–Dec aggregated over all years) for each pollutant.

    Parameters
    ----------
    series_dict : dict[str, pandas.DataFrame]
        DataFrames with columns ['date', 'value'].
    output_path : str
        PNG output file path.
    """
    n = len(series_dict)
    if n == 0:
        raise ValueError("No time series to plot.")

    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14, 4.5 * nrows),
        dpi=dpi,
    )
    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, (pollutant, df) in enumerate(series_dict.items()):
        ax = axes[i]

        df = df.copy()
        df["month_name"] = df["date"].dt.strftime("%b")
        df["month_num"] = df["date"].dt.month

        month_order = [
            name
            for _, name in sorted(zip(df["month_num"], df["month_name"]))
        ]
        month_order = list(dict.fromkeys(month_order))  # unique, keep order

        data_for_plot = [
            df.loc[df["month_name"] == m, "value"] for m in month_order
        ]

        ax.boxplot(data_for_plot, labels=month_order, patch_artist=True)
        ax.set_xlabel("Month")
        ax.set_ylabel(pollutant, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)

    for j in range(len(series_dict), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# 6. MAP WITH LABELS
# ---------------------------------------------------------------------
def create_map_with_labels(
    lat: float,
    lon: float,
    name: str,
    buffer_m: int = 500,
    country_name: str = "Iran",
    inset_dim: int = 120,
):
    """
    Create a satellite map around (lat, lon) with:
    - Buffer filled area
    - Red dot at the site
    - Grid & lat/lon labels
    - Scale bar + north arrow
    - Inset map for a country with red highlight around the site

    Returns
    -------
    PIL.Image.Image
    """
    point = ee.Geometry.Point([lon, lat])

    # Base Sentinel-2 image
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(point)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .first()
        .visualize(min=0, max=3000, bands=["B4", "B3", "B2"])
    )

    buffer_geom = point.buffer(buffer_m)
    filled_area = ee.Image().paint(buffer_geom, 1, 1).visualize(palette=["red"])
    dot = ee.Image().paint(point, 1, 80).visualize(palette=["red"])

    final_img = ee.ImageCollection([s2, filled_area, dot]).mosaic()

    region = point.buffer(10_000).bounds().getInfo()["coordinates"]
    params = {"region": region, "dimensions": 600, "format": "png"}
    url = final_img.getThumbURL(params)
    resp = requests.get(url)
    resp.raise_for_status()

    img = PILImage.open(BytesIO(resp.content)).convert("RGB")

    # Canvas with margins
    width, height = img.size
    margin_left, margin_right, margin_top, margin_bottom = 60, 60, 40, 40
    new_w = width + margin_left + margin_right
    new_h = height + margin_top + margin_bottom
    canvas = PILImage.new("RGB", (new_w, new_h), "white")
    canvas.paste(img, (margin_left, margin_top))
    draw = ImageDraw.Draw(canvas)

    # Fonts
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
        font_name = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        font_name = ImageFont.load_default()

    lon_min, lat_min = region[0][0]
    lon_max, lat_max = region[0][2]

    # Grid
    dash_len, gap_len = 3, 3
    lat_steps, lon_steps = 3, 3

    # Horizontal dashed lines
    for i in range(lat_steps + 1):
        y = margin_top + int(i * (height / lat_steps))
        for x_start in range(margin_left, margin_left + width, dash_len + gap_len):
            draw.line(
                [(x_start, y), (x_start + dash_len, y)],
                fill="gray",
                width=1,
            )

    # Vertical dashed lines
    for i in range(lon_steps + 1):
        x = margin_left + int(i * (width / lon_steps))
        for y_start in range(margin_top, margin_top + height, dash_len + gap_len):
            draw.line(
                [(x, y_start), (x, y_start + dash_len)],
                fill="gray",
                width=1,
            )

    # Latitude labels
    for i in range(lat_steps + 1):
        y = margin_top + int(i * (height / lat_steps))
        lat_val = lat_max - i * (lat_max - lat_min) / lat_steps
        draw.text((margin_left - 45, y - 11), f"{lat_val:.2f}", fill="black", font=font)
        draw.text(
            (margin_left + width + 10, y - 11),
            f"{lat_val:.2f}",
            fill="black",
            font=font,
        )

    # Longitude labels
    for i in range(lon_steps + 1):
        x = margin_left + int(i * (width / lon_steps))
        lon_val = lon_min + i * (lon_max - lon_min) / lon_steps
        draw.text((x - 18, margin_top - 25), f"{lon_val:.2f}", fill="black", font=font)
        draw.text(
            (x - 18, margin_top + height + 12),
            f"{lon_val:.2f}",
            fill="black",
            font=font,
        )

    # Company name near dot
    px_x, px_y = margin_left + width // 2, margin_top + height // 2
    draw.text((px_x + 10, px_y - 25), name, font=font_name, fill="black")

    # Scale bar (hard-coded 0–500 m visual)
    bar_w, bar_h = 100, 8
    sx = margin_left + 20
    sy = margin_top + height - 40
    draw.rectangle([sx, sy, sx + bar_w, sy + bar_h], fill="black")
    draw.text((sx, sy - 20), "0 m", fill="black", font=font)
    draw.text((sx + bar_w - 40, sy - 20), "500 m", fill="black", font=font)

    # North arrow
    ax_x, ax_y = margin_left + width - 40, margin_top + 40
    draw.line([(ax_x, ax_y), (ax_x, ax_y + 40)], fill="black", width=3)
    draw.polygon(
        [(ax_x - 6, ax_y + 10), (ax_x + 6, ax_y + 10), (ax_x, ax_y)],
        fill="black",
    )
    draw.text((ax_x - 8, ax_y - 20), "N", font=font_name, fill="black")

    # Inset country map
    country_fc = (
        ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
        .filter(ee.Filter.eq("country_na", country_name))
    )
    country_geom = country_fc.geometry()
    country_fill = ee.Image().paint(country_fc, 1).visualize(
        palette=["#DDDDDD"], forceRgbOutput=True
    )
    country_outline = ee.Image().paint(country_fc, 1, 2).visualize(
        palette=["black"], forceRgbOutput=True
    )
    buffer_30km = point.buffer(30_000)
    country_dot = ee.Image().paint(buffer_30km, 1).visualize(
        palette=["red"], forceRgbOutput=True
    )
    country_img = ee.ImageCollection(
        [country_fill, country_outline, country_dot]
    ).mosaic()
    country_region = country_geom.bounds().getInfo()["coordinates"]
    country_url = country_img.getThumbURL(
        {"region": country_region, "dimensions": 250, "format": "png"}
    )
    c_resp = requests.get(country_url)
    c_resp.raise_for_status()
    country_pil = PILImage.open(BytesIO(c_resp.content)).convert("RGB")

    inset_size = (int(inset_dim * 1.5), int(inset_dim * 1.5))
    country_pil = country_pil.resize(inset_size)
    inset_x = margin_left + width - inset_size[0]
    inset_y = margin_top + height - inset_size[1]
    canvas.paste(country_pil, (inset_x, inset_y))
    draw.rectangle(
        [inset_x, inset_y, inset_x + inset_size[0], inset_y + inset_size[1]],
        outline="black",
        width=2,
    )

    # Border
    draw.rectangle(
        [(margin_left, margin_top), (margin_left + width - 1, margin_top + height - 1)],
        outline="black",
        width=4,
    )

    return canvas


# ---------------------------------------------------------------------
# 7. PDF REPORT
# ---------------------------------------------------------------------
def build_company_pdf(
    excel_path: str,
    output_pdf: str,
    row_index: int = 0,
    time_series_image: str | None = None,
    monthly_boxplot_image: str | None = None,
):
    """
    Build a simple PDF report for one company/site.

    - Reads name/location/about text from Excel.
    - Generates a labeled map page.
    - Optionally adds:
      * Daily time series figure
      * Monthly boxplot figure

    Parameters
    ----------
    excel_path : str
        Excel file with at least columns: Name, latitude, longitude, About Company.
    output_pdf : str
        Output PDF filename.
    row_index : int
        0-based row index.
    time_series_image : str or None
        Path to a PNG with time series plots (optional).
    monthly_boxplot_image : str or None
        Path to a PNG with monthly boxplots (optional).
    """
    row, point, _ = load_site_from_excel(excel_path, row_index=row_index)
    company_name = str(row.get("Name", "Unknown")).strip()
    lat = float(row.get("latitude"))
    lon = float(row.get("longitude"))
    about = row.get("About Company", "No description available.")

    # ReportLab doc
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    story = []

    title_style = ParagraphStyle(
        name="Title",
        fontSize=24,
        alignment=TA_CENTER,
        leading=28,
        spaceAfter=20,
    )
    subtitle_style = ParagraphStyle(
        name="Subtitle",
        fontSize=12,
        leading=16,
        spaceAfter=10,
    )
    caption_style = ParagraphStyle(
        name="Caption",
        fontSize=10,
        leading=12,
        alignment=TA_CENTER,
        spaceAfter=20,
    )

    # --- Page 1: Company info + map ---
    story.append(Paragraph(f"<b>{company_name}</b>", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"About Company: {about}", subtitle_style))
    story.append(Spacer(1, 20))

    # Map
    map_img = create_map_with_labels(lat, lon, company_name)
    buf = BytesIO()
    map_img.save(buf, format="PNG")
    buf.seek(0)
    story.append(PDFImage(buf, width=400, height=300))
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Figure 1. Location of a petrochemical or power plant chimney.",
            caption_style,
        )
    )
    story.append(PageBreak())

    # --- Optional: daily time series ---
    if time_series_image and os.path.exists(time_series_image):
        story.append(Paragraph("Daily Time Series of Pollutants", title_style))
        story.append(Spacer(1, 20))
        story.append(PDFImage(time_series_image, width=480, height=360))
        story.append(PageBreak())

    # --- Optional: monthly boxplots ---
    if monthly_boxplot_image and os.path.exists(monthly_boxplot_image):
        story.append(Paragraph("Monthly Boxplots of Pollutants", title_style))
        story.append(Spacer(1, 20))
        story.append(PDFImage(monthly_boxplot_image, width=480, height=360))
        story.append(PageBreak())

    doc.build(story)
