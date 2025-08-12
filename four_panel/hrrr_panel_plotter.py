
'''
This is the 3rd edition (hopefully final) of a HRRR data pull & plot creation script. 

It: 

-identifies the most recent available HRRR run
-downloads GRIB files for the first hour of that run
-plots a variety of neat & useful meteorological analyses 
    -700mb RH analysis (w/ 500mb omega)
    -700mb wind/prate analysis
    -500mb analysis
    -accumulated precip
    -convective forecasting panels (in progress)
        -dewpt & sfc barbs
        -SBCAPE shaded & wind barbs for 0-6km shear 
    -total snowfall 
        -total snowfall downscaled w/ PRISM data? (proposed)
    -More ideas to come.
-deletes the first GRIBs, downloads the next model step's GRIBs, and plots for that next model step
-saves images in an organized fashion for later display on a web page

-is cool 
'''

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
import requests
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR   = Path("data")
IMAGE_DIR  = Path("images")
FCST_HOURS = list(range(1, 19))
LAT_BOUNDS = (35.0, 44.0)
LON_BOUNDS = (-116.2, -107.4)


# ──────────────────────────────────────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────────────────────────────────────

def identify_latest_run() -> datetime:
    """Return the most recent full-hour HRRR init (UTC)."""
    now = datetime.now(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)


def download_grib(fhr: int, init_time: datetime) -> dict:
    """
    Downloads all three GRIB2 files for a given forecast hour and returns a dict of paths.
    Keys: 'sfc', 'prs', 'bc' (actually subhf).
    """
    prefixes = {
        "sfc": "wrfsfcf",  # surface
        "prs": "wrfprsf",  # pressure-level
        "bc":  "wrfsubhf"  # sub-hourly file containing apcp (total precipitation)
    }

    date_str = init_time.strftime("%Y%m%d")
    hour_str = init_time.strftime("%H")
    fhr_str  = f"{fhr:02d}"

    filepaths = {}
    DATA_DIR.mkdir(exist_ok=True)

    for key, prefix in prefixes.items():
        fname   = f"hrrr.t{hour_str}z.{prefix}{fhr_str}.grib2"
        url     = f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date_str}/conus/{fname}"
        outpath = DATA_DIR / fname

        if outpath.exists():
            logger.info(f"Skip download (exists): {fname}")
            filepaths[key] = outpath
            continue

        logger.info(f"Downloading {fname}")
        try:
            resp = requests.get(url, stream=True, timeout=15)
            if resp.status_code == 404:
                logger.warning(f"404: {fname} not available")
                filepaths[key] = None
                continue
            resp.raise_for_status()

            with open(outpath, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            logger.info(f"✅ Saved: {outpath}")
            filepaths[key] = outpath

        except Exception as e:
            logger.warning(f"❌ Failed to download {fname}: {e}")
            filepaths[key] = None

    return filepaths


def safe_unlink(path: Path):
    """Try to delete a file, logging any errors."""
    if path is None:
        return
    try:
        path.unlink()
        logger.info(f"Deleted: {path.name}")
    except Exception as e:
        logger.warning(f"Failed to delete {path.name}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# PANEL PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def plot_panel_1(ds_prs, ds_sfc, fhr: int, init_time: datetime):
    lat_min, lat_max = 36.05, 42.45
    lon_min, lon_max = -114.55, -108.45

    t700  = ds_prs['t'].sel(isobaricInhPa=700) - 273.15
    prate = ds_sfc['prate'] * 3600  # mm/hr
    lats  = ds_prs['latitude'].values
    lons  = ds_prs['longitude'].values

    fig, ax = plt.subplots(
        figsize=(10, 8),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none', linewidth=0.5)

    PRECIP_COLORS = [
        "#f0f0f0", "#d9d9d9", "#bdbdbd", "#969696",
        "#737373", "#525252", "#252525", "#000000",
        "#111111", "#222222"
    ]
    precip_plot = ax.contourf(
        lons, lats, prate,
        levels=[0.1,0.5,1,2.5,5,10,20,30,40,50],
        colors=PRECIP_COLORS,
        transform=ccrs.PlateCarree(),
        alpha=0.7
    )
    cbar = fig.colorbar(precip_plot, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
    cbar.set_label('Precip Rate (mm/hr)')

    contours = ax.contour(
        lons, lats, t700,
        levels=np.arange(-18,23,1),
        cmap=plt.get_cmap("bwr"),
        linewidths=1.0,
        transform=ccrs.PlateCarree()
    )
    ax.clabel(contours, contours.levels, fmt='%d°C', inline=True, fontsize=10)

    zero_line = ax.contour(
        lons, lats, t700,
        levels=[0], colors='blue',
        linewidths=2.5, transform=ccrs.PlateCarree()
    )
    ax.clabel(zero_line, fmt='0°C', fontsize=9, inline=True)

    valid_time = init_time + timedelta(hours=fhr)
    ax.set_title(
        f"HRRR Init: {init_time:%Y-%m-%d %HZ} | Valid: {valid_time:%HZ}\n"
        "700 mb Isotherms + Instant Precip Rate",
        fontsize=11
    )

    out_path = IMAGE_DIR / f"{init_time:%Y%m%d%H}_f{fhr:02d}_panel1.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"✅ Saved Panel 1 image: {out_path}")


def plot_panel_2(ds_prs, fhr: int, init_time: datetime):
    lat_min, lat_max = 36.05, 42.45
    lon_min, lon_max = -114.55, -108.45

    rh700      = ds_prs["r"].sel(isobaricInhPa=700)
    u700       = ds_prs["u"].sel(isobaricInhPa=700)
    v700       = ds_prs["v"].sel(isobaricInhPa=700)
    omega500   = ds_prs["w"].sel(isobaricInhPa=500)

    lats  = ds_prs["latitude"].values
    lons  = ds_prs["longitude"].values

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none', linewidth=0.5)

    ax.contour(
        lons, lats, omega500,
        levels=np.arange(-30,0,2),
        colors='blue', linestyles='dashed',
        linewidths=1.0, transform=ccrs.PlateCarree()
    )
    ax.contour(
        lons, lats, omega500,
        levels=np.arange(2,10,2),
        colors='red', linestyles='solid',
        linewidths=1.0, transform=ccrs.PlateCarree()
    )

    rh_plot = ax.contourf(
        lons, lats, rh700,
        levels=np.arange(0,105,10),
        cmap="BrBG",
        alpha=0.8,
        transform=ccrs.PlateCarree()
    )

    skip = (slice(None, None, 10), slice(None, None, 10))
    ax.barbs(
        lons[skip], lats[skip],
        u700.values[skip], v700.values[skip],
        length=5, linewidth=0.6,
        transform=ccrs.PlateCarree()
    )

    valid_time = init_time + timedelta(hours=fhr)
    ax.set_title(
        f"HRRR Init: {init_time:%Y-%m-%d %HZ} | Valid: {valid_time:%HZ}\n"
        "700mb RH + Winds + 500mb Omega",
        fontsize=11
    )
    cbar = fig.colorbar(rh_plot, ax=ax, orientation="vertical", shrink=0.7, pad=0.02)
    cbar.set_label("700mb RH (%)")

    out_path = IMAGE_DIR / f"{init_time:%Y%m%d%H}_f{fhr:02d}_panel2.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"✅ Saved Panel 2 image: {out_path}")


def plot_panel_3(ds_prs, fhr: int, init_time: datetime):
    lat_min, lat_max = 25.0, 50.0
    lon_min, lon_max = -135.0, -100.0

    hgt500   = ds_prs["gh"].sel(isobaricInhPa=500) / 10
    vort500  = ds_prs["absv"].sel(isobaricInhPa=500) * 1e5
    u500     = ds_prs["u"].sel(isobaricInhPa=500)
    v500     = ds_prs["v"].sel(isobaricInhPa=500)

    lats = ds_prs["latitude"].values
    lons = ds_prs["longitude"].values

    fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none', linewidth=0.5)

    ax.contourf(
        lons, lats, vort500,
        levels=np.arange(-5,65,5),
        cmap="plasma", extend="max",
        transform=ccrs.PlateCarree(), alpha=0.8
    )

    hgt_contours = ax.contour(
        lons, lats, hgt500,
        levels=np.arange(480,600,4),
        colors="black", linewidths=2.0,
        transform=ccrs.PlateCarree()
    )
    ax.clabel(hgt_contours, fmt='%d', fontsize=9)

    skip = (slice(None, None, 40), slice(None, None, 40))
    ax.barbs(
        lons[skip], lats[skip],
        u500.values[skip], v500.values[skip],
        length=6, linewidth=0.7,
        transform=ccrs.PlateCarree()
    )

    valid_time = init_time + timedelta(hours=fhr)
    ax.set_title(
        f"HRRR Init: {init_time:%Y-%m-%d %HZ} | Valid: {valid_time:%HZ}\n"
        "500mb Heights, Vorticity, Winds",
        fontsize=11
    )
    cbar = fig.colorbar(ax.collections[0], ax=ax, orientation="vertical", shrink=0.7, pad=0.02)
    cbar.set_label("500mb Abs Vort (1e-5 s⁻¹)")

    out_path = IMAGE_DIR / f"{init_time:%Y%m%d%H}_f{fhr:02d}_panel3.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"✅ Saved Panel 3 image: {out_path}")


def plot_panel_4(ds_bc, fhr: int, init_time: datetime):
    """Accumulated precip panel using 'tp' from the subhf file."""
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # 'tp' is in meters by default in CF; convert to inches
    acpc_m  = ds_bc['tp'].isel(step=0)    # or .isel(time=0) depending on your cfgrib version
    acpc_mm = acpc_m * 1000.0
    acpc_in = acpc_mm / 25.4

    lons = ds_bc['longitude'].values
    lats = ds_bc['latitude'].values

    pcm = ax.pcolormesh(
        lons, lats, acpc_in,
        cmap='Blues', shading='auto',
        transform=ccrs.PlateCarree()
    )

    ax.set_extent([LON_BOUNDS[0], LON_BOUNDS[1], LAT_BOUNDS[0], LAT_BOUNDS[1]])
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3)

    valid_time = init_time + timedelta(hours=fhr)
    ax.set_title(
        f"HRRR Init: {init_time:%Y-%m-%d %HZ} | Valid: {valid_time:%HZ}\n"
        "Accumulated QPF (inches)",
        fontsize=11
    )

    cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_label('inches')

    out_path = IMAGE_DIR / f"{init_time:%Y%m%d%H}_f{fhr:02d}_panel4.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"✅ Saved Panel 4 image: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

def main():
    init_time = identify_latest_run()
    IMAGE_DIR.mkdir(exist_ok=True)

    for fhr in FCST_HOURS:
        logger.info(f"Processing f{fhr:02d}")

        paths = download_grib(fhr, init_time)
        sfc_p = paths["sfc"]
        prs_p = paths["prs"]
        bc_p  = paths["bc"]

        if not sfc_p or not prs_p or not bc_p:
            logger.warning(f"Skipping f{fhr:02d} (one or more files missing)")
            continue

        try:
            ds_prs = xr.open_dataset(prs_p, engine="cfgrib",
                                     filter_by_keys={"typeOfLevel": "isobaricInhPa"})
            ds_sfc = xr.open_dataset(sfc_p, engine="cfgrib",
                                     filter_by_keys={"typeOfLevel": "surface", "stepType": "instant"})
            # only grab the surface‐level messages (which includes 'tp')
            ds_bc = xr.open_dataset(
                bc_p,
                engine="cfgrib",
                filter_by_keys={"typeOfLevel": "surface", "stepType": "accum",}
            )


            plot_panel_1(ds_prs, ds_sfc, fhr, init_time)
            plot_panel_2(ds_prs, fhr, init_time)
            plot_panel_3(ds_prs, fhr, init_time)

            if fhr >= 2:
                plot_panel_4(ds_bc, fhr, init_time)

            ds_prs.close()
            ds_sfc.close()
            ds_bc.close()

        except Exception as e:
            logger.error(f"Failed to parse or plot f{fhr:02d}: {e}")
            continue

        # clean up
        safe_unlink(sfc_p)
        safe_unlink(prs_p)
        safe_unlink(bc_p)


if __name__ == "__main__":
    main()
