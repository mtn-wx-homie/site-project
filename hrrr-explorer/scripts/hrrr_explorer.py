"""
hrrr_explorer_1.py - Interactive HRRR model explorer for Utah (using cfgrib)

Features:
- Download HRRR GRIB2 files for specified date, run hour, and forecast lead times
- Trim to Utah domain
- Extract 700 mb temperature and instantaneous precipitation rate
- Plot each field on a map and save as images
- Interactive viewer: navigate frames with arrow keys

Run command format:
python3 hrrr_explorer_test.py --date YYYYMMDD --run_hour HH --hours H1 H2 H3 ... [--browse]
"""

import os
import glob
import argparse
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import requests
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore", category=FutureWarning)

# Utah domain extents with ~30 mile buffer
LAT_MIN, LAT_MAX = 36.05, 42.45
LON_MIN, LON_MAX = -114.55, -108.45

# Custom colormap and precip levels for WeatherBell style
TEMP_CMAP = plt.get_cmap("RdYlBu_r")
PRECIP_LEVELS = [0.1, 0.5, 1, 2.5, 5, 10, 20, 30, 40, 50]  # mm/hr
PRECIP_COLORS = [
    "#f0f0f0", "#d9d9d9", "#bdbdbd", "#969696",
    "#737373", "#525252", "#252525", "#000000",
    "#111111", "#222222"
]


def browse_plots(plot_dir):
    """Open an interactive Matplotlib window to page through saved PNGs."""
    files = sorted(glob.glob(os.path.join(plot_dir, "*.png")))
    if not files:
        print(f"No images found in {plot_dir}")
        return

    idx = 0
    fig, ax = plt.subplots()
    im = ax.imshow(mpimg.imread(files[idx]))
    ax.set_axis_off()
    fig.suptitle(os.path.basename(files[idx]), fontsize=10)

    def on_key(event):
        nonlocal idx
        if event.key in ['right', 'd']:
            idx = (idx + 1) % len(files)
        elif event.key in ['left', 'a']:
            idx = (idx - 1) % len(files)
        else:
            return
        im.set_data(mpimg.imread(files[idx]))
        fig.suptitle(os.path.basename(files[idx]), fontsize=10)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def download_hrrr(date_str, run_hour, forecast_hour, output_dir):
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


def process_grib(grib_paths, plot_dir):
    try:
        ds_temp = xr.open_dataset(
            grib_paths['prs'],
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}}
        )
        temp = ds_temp['t'].sel(isobaricInhPa=700) - 273.15
        lats = ds_temp['latitude'].values
        lons = ds_temp['longitude'].values
        init_time = np.datetime64(ds_temp.time.values)
        valid_time = np.datetime64(ds_temp.valid_time.values)

        ds_precip = xr.open_dataset(
            grib_paths['sfc'],
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface", "stepType": "instant"}}
        )
        precip = ds_precip['prate'] * 3600  # kg/m²/s → mm/hr

    except Exception as e:
        print(f"  ❌ GRIB processing failed: {e}")
        return

    fig, ax = plt.subplots(
        figsize=(10, 8),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none', linewidth=0.5)

    precip_plot = ax.contourf(
        lons, lats, precip,
        levels=PRECIP_LEVELS, colors=PRECIP_COLORS,
        transform=ccrs.PlateCarree(), alpha=0.7
    )
    cbar = fig.colorbar(
        precip_plot, ax=ax,
        orientation='vertical', shrink=0.7, pad=0.02
    )
    cbar.set_label('Precip Rate (mm/hr)')

    temp_levels = np.arange(-18, 23, 1)
    contours = ax.contour(
        lons, lats, temp,
        levels=temp_levels, cmap=TEMP_CMAP,
        linewidths=1.0, transform=ccrs.PlateCarree()
    )
    ax.clabel(contours, fmt='%d°C', inline=True, fontsize=8)

    init_dt = datetime.fromtimestamp(
        init_time.astype('datetime64[s]').astype(int),
        tz=timezone.utc
    )
    valid_dt = datetime.fromtimestamp(
        valid_time.astype('datetime64[s]').astype(int),
        tz=timezone.utc
    )
    init_str = init_dt.strftime('%Y-%m-%d %HZ')
    valid_str = valid_dt.strftime('%HZ')
    ax.set_title(
        f"HRRR Init: {init_str} | Valid: {valid_str} | 700 mb Isotherms",
        fontsize=12
    )

    out_fn = f"plot_{init_str.replace(':','').replace(' ','_')}_{valid_str}.png"
    out_path = os.path.join(plot_dir, out_fn)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date", type=str, required=True,
        help="Initialization date (YYYYMMDD)"
    )
    parser.add_argument(
        "--run_hour", type=int, required=True,
        help="HRRR init hour (0–23)"
    )
    parser.add_argument(
        "--hours", type=int, nargs="+", required=True,
        help="Forecast lead times to process (e.g. 0 6 12)"
    )
    parser.add_argument(
        "--plot_dir", type=str, default="plots",
        help="Directory to save plot images"
    )
    parser.add_argument(
        "--browse", action="store_true",
        help="Open an interactive window to page through all plots"
    )
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

        # Clear out old plots before this run
    for file in os.listdir(args.plot_dir):
        if file.endswith(".png"):
            os.remove(os.path.join(args.plot_dir, file))
    print(f"🧹 Cleared old .png plots from {args.plot_dir}")


    for fxx in args.hours:
        paths = download_hrrr(args.date, args.run_hour, fxx, "data")
        if paths.get('prs') and paths.get('sfc'):
            process_grib(paths, args.plot_dir)

    if args.browse:
        print("🕹️  browse flag detected, launching viewer…")
        browse_plots(args.plot_dir)

if __name__ == "__main__":
    main()
