
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Utah domain bounds (lat_min, lat_max), (lon_min, lon_max)
lat_bounds = (35.0, 44.0)
lon_bounds = (-116.2, -107.4)

# Forecast hours
fcst_hours = list(range(1, 19))

# Directories
DATA_DIR  = Path("data")
IMAGE_DIR = Path("images")


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def identify_last_hrrr() -> datetime:
    """Return the most recent completed HRRR init in UTC (hourly)."""
    now = datetime.now(timezone.utc)
    last_hr = now.replace(minute=0, second=0, microsecond=0)
    return last_hr - timedelta(hours=1)

init_time = identify_last_hrrr()

def download_hrrr(
    fhr: int,
    save_dir: Path = DATA_DIR,
    file_type: str = "sfc"
) -> Path:
    """
    Download one HRRR GRIB2 for forecast hour fhr.
    file_type="sfc" -> wrfsfcf, "prs" -> wrfprsf.
    """
    prefixes = {"sfc": "wrfsfcf", "prs": "wrfprsf", "subhf": "wrfsubhf"}
    if file_type not in prefixes:
        raise ValueError("file_type must be 'sfc', 'prs', or subhf")
    date_str = init_time.strftime("%Y%m%d")
    hour_str = init_time.strftime("%H")
    fhr_str  = f"{fhr:02d}"
    suffix   = prefixes[file_type]
    fname    = f"hrrr.t{hour_str}z.{suffix}{fhr_str}.grib2"

    save_dir.mkdir(parents=True, exist_ok=True)
    outpath = save_dir / fname
    if outpath.exists():
        logger.info(f"Skip download (exists): {fname}")
        return outpath

    url = f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date_str}/conus/{fname}"
    logger.info(f"Downloading {fname}")
    resp = requests.get(url, stream=True, timeout=10)
    if resp.status_code == 404:
        logger.warning(f"404 not found: {fname}")
        return outpath
    resp.raise_for_status()
    with open(outpath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    logger.info(f"Saved {outpath}")

    grib_subhf = download_hrrr(fhr, save_dir, file_type="subhf")
    ds_subhf = xr.open_dataset(
    grib_subhf,
    engine="cfgrib",
    filter_by_keys={"typeOfLevel": "surface", "stepType": "accumulation"}
)
    print(ds_subhf.data_vars)



    return outpath


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

#configuring how our plots will look (for panel 1)
PRECIP_LEVELS = [0.1, 0.5, 1, 2.5, 5, 10, 20, 30, 40, 50]
PRECIP_COLORS = [
    "#f0f0f0", "#d9d9d9", "#bdbdbd", "#969696",
    "#737373", "#525252", "#252525", "#000000",
    "#111111", "#222222"
]
TEMP_LEVELS = np.arange(-18, 23, 1)


def process_hrrr(
    lat_bounds, lon_bounds, fcst_hours,
    data_dir: Path = DATA_DIR,
    image_dir: Path = IMAGE_DIR
):
    date_str = init_time.strftime("%Y%m%d")
    hour_str = init_time.strftime("%H")
    image_dir.mkdir(parents=True, exist_ok=True)
    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds

    for fhr in fcst_hours:
        fhr_str = f"{fhr:02d}"

        # Download GRIBs
        grib_sfc = download_hrrr(fhr, data_dir, file_type="sfc")
        grib_prs = download_hrrr(fhr, data_dir, file_type="prs")
        if not (grib_sfc.exists() and grib_prs.exists()):
            logger.warning(f"Skipping f{fhr_str}, missing files")
            continue

        try:
            # Open datasets
            ds_prs = xr.open_dataset(
                grib_prs, engine="cfgrib",
                filter_by_keys={"typeOfLevel": "isobaricInhPa"}
            )
            ds_sfc_inst = xr.open_dataset(
                grib_sfc, engine="cfgrib",
                filter_by_keys={"typeOfLevel": "surface", "stepType": "instant"}
            )

            # Extract fields (these are for panel1)
            t700 = ds_prs["t"].sel(isobaricInhPa=700) - 273.15  # K → °C
            prate = ds_sfc_inst["prate"] * 3600  # kg/m²/s → mm/hr
            lats = ds_prs["latitude"].values
            lons = ds_prs["longitude"].values

        except Exception as e:
            logger.error(f"GRIB parse failed for f{fhr_str}: {e}")
            continue



        # Plot
        fig, ax = plt.subplots(
            figsize=(10, 8),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])
        ax.coastlines(resolution='10m', linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none', linewidth=0.5)

        # Precip shading (grayscale)
        precip_plot = ax.contourf(
            lons, lats, prate,
            levels=PRECIP_LEVELS, colors=PRECIP_COLORS,
            transform=ccrs.PlateCarree(), alpha=0.7
        )

        # Black 700mb isotherms
        temp_contours = ax.contour(
            lons, lats, t700,
            levels=TEMP_LEVELS,
            colors="black",
            linewidths=1.5,
            transform=ccrs.PlateCarree()
        )
        ax.clabel(temp_contours, temp_contours.levels, fmt="%d°C", fontsize=8, inline=True, inline_spacing=0)

        # Highlight bold 0°C in blue
        zero_line = ax.contour(
            lons, lats, t700,
            levels=[0],
            colors='blue',
            linewidths=2.5,
            transform=ccrs.PlateCarree()
        )
        ax.clabel(zero_line, fmt='0°C', fontsize=9, inline=True)

        # Title and colorbar
        init_str = init_time.strftime("%Y-%m-%d %HZ")
        valid_time = init_time + timedelta(hours=fhr)
        valid_str = valid_time.strftime("%HZ")
        ax.set_title(
            f"HRRR Init: {init_str} | Valid: {valid_str}\n700 mb Isotherms + Precip Rate",
            fontsize=11
        )
        cbar = fig.colorbar(precip_plot, ax=ax, orientation="vertical", shrink=0.7, pad=0.02)
        cbar.set_label("Precip Rate (mm/hr)")

        # Save
        outpath = image_dir / f"{date_str}{hour_str}_f{fhr_str}_panel1.png"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"✅ Saved Panel 1 image: {outpath}")

        ds_sfc_acc = xr.open_dataset(
            grib_sfc,
            engine="cfgrib",
            filter_by_keys={"typeOfLevel": "surface", "stepType": "accumulation"}
        )
        print(ds_sfc_acc.data_vars)

        # ───── PANEL 2: 700mb RH + Wind + 500mb Omega ─────
        try:
            rh700 = ds_prs["r"].sel(isobaricInhPa=700)
            u700 = ds_prs["u"].sel(isobaricInhPa=700)
            v700 = ds_prs["v"].sel(isobaricInhPa=700)
            omega500 = ds_prs["w"].sel(isobaricInhPa=500)  # Units: Pa/s

            fig2, ax2 = plt.subplots(
                figsize=(10, 8),
                subplot_kw={"projection": ccrs.PlateCarree()}
            )
            ax2.set_extent([lon_min, lon_max, lat_min, lat_max])
            ax2.coastlines(resolution='10m', linewidth=0.5)
            ax2.add_feature(cfeature.STATES, linewidth=0.5)
            ax2.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none', linewidth=0.5)

            # Lightly shaded omega (ascending = negative Pa/s)
            omega_levels_neg = np.arange(-30, 0, 2)  # ascent
            omega_levels_pos = np.arange(2, 10, 2)   # descent

            # Ascent contours (dashed blue)
            ascent = ax2.contour(
                lons, lats, omega500,
                levels=omega_levels_neg,
                colors='blue',
                linestyles='dashed',
                linewidths=1.0,
                transform=ccrs.PlateCarree()
            )

            # Descent contours (solid red)
            descent = ax2.contour(
                lons, lats, omega500,
                levels=omega_levels_pos,
                colors='red',
                linestyles='solid',
                linewidths=1.0,
                transform=ccrs.PlateCarree()
            )
            ax2.clabel(ascent, fmt='%d', fontsize=7)
            ax2.clabel(descent, fmt='%d', fontsize=7)


            # RH shading
            rh_levels = np.arange(0, 105, 10)
            rh_plot = ax2.contourf(
                lons, lats, rh700,
                levels=rh_levels,
                cmap="BrBG",
                alpha=0.8,
                transform=ccrs.PlateCarree()
            )

            # Wind barbs at 700mb
            skip = (slice(None, None, 10), slice(None, None, 10))
            ax2.barbs(
                lons[skip], lats[skip],
                u700.values[skip], v700.values[skip],
                length=5, linewidth=0.6, transform=ccrs.PlateCarree()
            )

            # Title + colorbar
            ax2.set_title(
                f"HRRR Init: {init_str} | Valid: {valid_str}\n700mb RH + Winds + 500mb Omega",
                fontsize=11
            )
            cbar = fig2.colorbar(rh_plot, ax=ax2, orientation="vertical", shrink=0.7, pad=0.02)
            cbar.set_label("700mb RH (%)")

            # Save figure
            outpath2 = image_dir / f"{date_str}{hour_str}_f{fhr_str}_panel2.png"
            fig2.savefig(outpath2, dpi=150, bbox_inches="tight")
            plt.close(fig2)
            logger.info(f"✅ Saved Panel 2 image: {outpath2}")

        except Exception as e:
            logger.warning(f"⚠️ Panel 2 failed at f{fhr_str}: {e}")

                # Assume 500mb fields already sliced:
        z500 = ds_prs['gh'].sel(isobaricInhPa=500) / 10  # convert to dam
        vo500 = ds_prs['absv'].sel(isobaricInhPa=500) * 1e5  # to 1e-5 s^-1
        u500 = ds_prs['u'].sel(isobaricInhPa=500)
        v500 = ds_prs['v'].sel(isobaricInhPa=500)
        wind_speed = np.sqrt(u500**2 + v500**2)

        # Define lat/lon grid
        lons, lats = ds_prs['longitude'], ds_prs['latitude']

                # ───── PANEL 3: 500mb Heights + Vorticity + Wind Barbs ─────
        try:
            lat_min_p3, lat_max_p3 = 28, 50
            lon_min_p3, lon_max_p3 = -125, -102
            fig3, ax3 = plt.subplots(
                figsize=(10, 8),
                subplot_kw={"projection": ccrs.PlateCarree()}
            )
            ax3.set_extent([lon_min_p3, lon_max_p3, lat_min_p3, lat_max_p3])
            ax3.coastlines(resolution='10m', linewidth=0.5)
            ax3.add_feature(cfeature.STATES, linewidth=0.5)
            ax3.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none', linewidth=0.5)

            # Fields
            z500 = ds_prs['gh'].sel(isobaricInhPa=500) / 10  # dam
            vo500 = ds_prs['absv'].sel(isobaricInhPa=500) * 1e5  # 1e-5 s^-1
            u500 = ds_prs['u'].sel(isobaricInhPa=500)
            v500 = ds_prs['v'].sel(isobaricInhPa=500)
            wind_speed = np.sqrt(u500**2 + v500**2)

            # Shading: vorticity
            vort_levels = np.arange(-4, 40, 4)
            vort_cmap = plt.get_cmap("OrRd")
            vort_plot = ax3.contourf(
                lons, lats, vo500,
                levels=vort_levels,
                cmap=vort_cmap,
                extend="both",
                alpha=0.6,
                transform=ccrs.PlateCarree()
            )

            # Contours: geopotential height
            height_levels = np.arange(480, 600, 6)
            height_contours = ax3.contour(
                lons, lats, z500,
                levels=height_levels,
                colors="black",
                linewidths=1.0,
                transform=ccrs.PlateCarree()
            )
            ax3.clabel(height_contours, fontsize=8, inline=True)

            # Wind barbs: greyscale logic
            stride = 10
            lons_b = lons[::stride, ::stride]
            lats_b = lats[::stride, ::stride]
            u_b = u500[::stride, ::stride]
            v_b = v500[::stride, ::stride]
            wspd_b = wind_speed[::stride, ::stride] * 1.94384  # m/s to knots

            for i in range(lons_b.shape[0]):
                for j in range(lons_b.shape[1]):
                    wspd = wspd_b[i, j].item()
                    color = "black" if wspd >= 50 else "lightgray"
                    ax3.barbs(
                        lons_b[i, j].item(), lats_b[i, j].item(),
                        u_b[i, j].item(), v_b[i, j].item(),
                        length=5.5,
                        color=color,
                        linewidth=0.6,
                        transform=ccrs.PlateCarree()
                    )

            ax3.set_title(
                f"HRRR Init: {init_str} | Valid: {valid_str}\n500mb Heights, Vorticity, Winds",
                fontsize=11
            )

            cbar = fig3.colorbar(vort_plot, ax=ax3, orientation="vertical", shrink=0.7, pad=0.02)
            cbar.set_label("500mb Absolute Vorticity (1e-5 s⁻¹)")

            # Save figure
            outpath3 = image_dir / f"{date_str}{hour_str}_f{fhr_str}_panel3.png"
            fig3.savefig(outpath3, dpi=150, bbox_inches="tight")
            plt.close(fig3)
            logger.info(f"✅ Saved Panel 3 image: {outpath3}")

        except Exception as e:
            logger.warning(f"⚠️ Panel 3 failed at f{fhr_str}: {e}")




        ds_prs.close()
        ds_sfc_inst.close()
        grib_sfc.unlink()
        grib_prs.unlink()

if __name__ == "__main__":
    process_hrrr(lat_bounds, lon_bounds, fcst_hours)

