#!/usr/bin/env python3
"""
four_panel.py: Auto-generate HRRR four-panel plots for all forecast hours

This script:
  - Assumes the latest available HRRR initialization at UTC now minus 1 hour
  - Loops through all forecast hours (0–18)
  - Downloads one GRIB2 file at a time using the same download_hrrr helper as hrrr_explorer.py
  - Extracts needed fields, deletes file immediately
  - Creates four-panel plots (700mb temp, 700mb RH, 700mb winds & omega, surface precip)
  - Saves each image to `images/hrrr_<YYYYMMDDHH>_f<FFF>.png`
  - Uses minimal RAM via immediate file cleanup and garbage collection

Usage:
  cd four_panel
  python four_panel.py

No arguments needed. Outputs drop into `four_panel/images/`.
"""

import os
import tempfile
import requests
from requests.exceptions import HTTPError
import pygrib
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import gc
from datetime import datetime, timedelta, timezone

# Setup constants and directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)
TZ = timezone.utc
FORECAST_HOURS = range(0, 19)


def download_hrrr(date_str, run_hour, forecast_hour, output_dir):
    """Download HRRR surface and pressure files for a given forecast hour."""
    out_paths = {}
    for level_type in ['wrfsfcf', 'wrfprsf']:
        filetype = 'sfc' if level_type == 'wrfsfcf' else 'prs'
        url = (
            f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/"
            f"hrrr.{date_str}/conus/"
            f"hrrr.t{run_hour:02d}z.{level_type}{forecast_hour:02d}.grib2"
        )
        out_path = os.path.join(
            output_dir,
            f"hrrr_{date_str}_{run_hour:02d}_f{forecast_hour:02d}_{filetype}.grib2"
        )

        if not os.path.exists(out_path):
            print(f"Downloading {url}")
            try:
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Saved to {out_path}")
            except Exception as e:
                print(f"  ❌ Download failed ({filetype}): {e}")
                out_path = None
        else:
            print(f"Exists: {out_path}")

        out_paths[filetype] = out_path
    return out_paths


def extract_field(grib_path, name, level_type=None, level=None):
    grbs = pygrib.open(grib_path)
    try:
        if level_type and level is not None:
            msg = grbs.select(name=name, typeOfLevel=level_type, level=level)[0]
        else:
            msg = grbs.select(name=name)[0]
    finally:
        grbs.close()
    data = msg.values
    lats, lons = msg.latlons()
    os.remove(grib_path)
    return data, lats, lons


def plot_and_save(fields, lats, lons, date_str, run_hour, fh):
    t700, rh700, uw700, vw700, omega, precip = fields
    fig = plt.figure(figsize=(14, 10))
    proj = ccrs.PlateCarree()
    axs = [fig.add_subplot(2, 2, i+1, projection=proj) for i in range(4)]

    # Panel 1: 700mb Temp
    cs = axs[0].contour(lons, lats, t700 - 273.15, levels=np.arange(-40, 30, 5))
    axs[0].clabel(cs, inline=1, fontsize=8); axs[0].set_title('700mb Temp (°C)')
    # Panel 2: 700mb RH
    cf1 = axs[1].contourf(lons, lats, rh700, levels=np.arange(0, 101, 10), cmap='Greens')
    fig.colorbar(cf1, ax=axs[1], orientation='horizontal'); axs[1].set_title('700mb RH (%)')
    # Panel 3: Winds & Omega
    axs[2].barbs(lons[::5, ::5], lats[::5, ::5], uw700[::5, ::5], vw700[::5, ::5])
    cf2 = axs[2].contourf(lons, lats, omega, cmap='coolwarm', alpha=0.6)
    fig.colorbar(cf2, ax=axs[2], orientation='vertical'); axs[2].set_title('700mb Winds & Omega')
    # Panel 4: 1hr Precip
    cf3 = axs[3].contourf(lons, lats, precip, levels=np.arange(0, 10.1, 0.5))
    fig.colorbar(cf3, ax=axs[3], orientation='horizontal'); axs[3].set_title('1hr Precip (mm)')

    plt.tight_layout()
    fname = f"hrrr_{date_str}{run_hour:02d}_f{fh:03d}.png"
    path = os.path.join(IMAGES_DIR, fname)
    plt.savefig(path, dpi=150); plt.close(fig); gc.collect()
    print(f"Saved plot to {path}")


def main():
    now = datetime.now(TZ).replace(minute=0, second=0, microsecond=0)
    init = now - timedelta(hours=1)
    date_str = init.strftime('%Y%m%d')
    run_hour = init.hour
    print(f"Using init {date_str}{run_hour:02d} UTC")

    for fh in FORECAST_HOURS:
        paths = download_hrrr(date_str, run_hour, fh, BASE_DIR)
        if not paths.get('sfc') or not paths.get('prs'):
            continue
        # Extract fields
        precip, lats, lons = extract_field(paths['sfc'], 'Total Precipitation')
        t700, _, _ = extract_field(paths['prs'], 'Temperature', 'isobaricInhPa', 700)
        rh700, _, _ = extract_field(paths['prs'], 'Relative humidity', 'isobaricInhPa', 700)
        uw700, _, _ = extract_field(paths['prs'], 'U component of wind', 'isobaricInhPa', 700)
        vw700, _, _ = extract_field(paths['prs'], 'V component of wind', 'isobaricInhPa', 700)
        omega, _, _ = extract_field(paths['prs'], 'Vertical velocity', 'isobaricInhPa', 700)
        plot_and_save((t700, rh700, uw700, vw700, omega, precip), lats, lons, date_str, run_hour, fh)

if __name__ == '__main__':
    main()
