
#largely discontinued, decided to build individual four_panels for each model i want to work with. 



"""
four_panel.py - Download & process the most recent two runs of HRRR and GFS (ECMWF disabled by default).

This script:
1. Computes the two latest initialization times for each enabled model.
2. For each init and forecast hour:
   - Downloads the GRIB2 file(s) into `model_data/` (caching existing files).
   - Opens data via cfgrib with appropriate filters.
   - Extracts fields: 700 mb winds/temp/RH, surface precipitation rate, differential vorticity advection.
   - Plots and saves four .pngs per frame into `images/YYYYMMDDHH/`.
   - Deletes GRIB2 files after processing (comment out to cache).

Dependencies:
    pip install numpy xarray cfgrib requests matplotlib cartopy scipy
"""
import os
from datetime import datetime, timezone, timedelta
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import requests
import gc

# --- CONFIG ---
VERBOSE = True
PLOT_PANELS = [1, 2, 3, 4]

MODEL_CONFIG = {
    'hrrr':  {'interval': 1,  'forecast_hours': range(1, 19),    'enabled': True},
    'gfs':   {'interval': 6,  'forecast_hours': range(0, 385, 3), 'enabled': True},
    'ecmwf': {'interval': 12, 'forecast_hours': range(0, 241, 3), 'enabled': False}
}

# Domain extents
LAT_MIN, LAT_MAX = 36.05, 42.45
LON_MIN, LON_MAX = -114.55, -108.45

# Natural Earth features
COAST = cfeature.NaturalEarthFeature('physical', 'coastline', '50m', edgecolor='black', facecolor='none')
STATES = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lakes', '50m', edgecolor='gray', facecolor='none')

# Directories
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'model_data')
plot_root = os.path.join(script_dir, 'images')

# Logging helper

def log(msg):
    if VERBOSE:
        print(msg)

# Compute latest inits
def get_latest_inits(interval, count=2, lag=0):
    now = datetime.now(timezone.utc) - timedelta(hours=lag)
    now = now.replace(minute=0, second=0, microsecond=0)
    last_hour = (now.hour // interval) * interval
    latest = now.replace(hour=last_hour)
    return [(latest - timedelta(hours=i*interval)).strftime('%Y%m%d%H') for i in range(count)]

# Download GRIB2
def download_grib(model, init, fxx):
    date, hour = init[:8], init[8:]
    os.makedirs(data_dir, exist_ok=True)
    def fetch(url, path):
        if os.path.exists(path):
            log(f"✅ Cached {path}")
            return True
        try:
            log(f"Downloading {url}")
            r = requests.get(url, stream=True); r.raise_for_status()
            with open(path,'wb') as f:
                for chunk in r.iter_content(8192): f.write(chunk)
            return True
        except requests.HTTPError:
            log(f"⚠️ Missing {url}")
            return False

    if model == 'hrrr':
        sfc = os.path.join(data_dir, f'hrrr_{init}_f{fxx:02d}_sfc.grib2')
        prs = os.path.join(data_dir, f'hrrr_{init}_f{fxx:02d}_prs.grib2')
        url_sfc = f'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date}/conus/hrrr.t{hour}z.wrfsfcf{fxx:02d}.grib2'
        url_prs = f'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date}/conus/hrrr.t{hour}z.wrfprsf{fxx:02d}.grib2'
        ok1 = fetch(url_sfc, sfc)
        ok2 = fetch(url_prs, prs)
        return {'sfc': sfc, 'prs': prs} if ok1 and ok2 else None
    if model == 'gfs':
        path = os.path.join(data_dir, f'gfs_{init}_f{fxx:03d}.grib2')
        url = f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{date}/{hour}/atmos/gfs.t{hour}z.pgrb2.0p25.f{fxx:03d}'
        if fetch(url, path): return path
    return None

# Squeeze to 2D array
def ensure_2d(da):
    if da is None: return None
    da = da.squeeze()
    if da.ndim > 2:
        idx = {d:0 for d in da.dims[:-2]}
        da = da.isel(idx)
    return da

# Load variable via cfgrib
def load_var(name, path, level=None):
    keys = {'typeOfLevel':'isobaricInhPa', 'shortName':name}
    if level: keys['level'] = level
    try:
        ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs={'filter_by_keys':keys})
        da = ds[name].load(); ds.close()
        return ensure_2d(da)
    except Exception as e:
        log(f"⚠️ Missing {name} @ level {level}: {e}")
        return None

# Format title
def format_title(init, fxx):
    idt = datetime.strptime(init,'%Y%m%d%H')
    vdt = idt + timedelta(hours=fxx)
    return f"Init:{idt:%Y-%m-%d%HZ} Valid:{vdt:%Y-%m-%d%HZ}"

# Process and plot a frame
def process_frame(model, paths, init, fxx):
    log(f"--- {model.upper()} {init}+{fxx} ---")
    if not paths: return
    sfc = paths['sfc'] if isinstance(paths,dict) else paths
    prs = paths['prs'] if isinstance(paths,dict) else paths

    # Load fields
    u700  = load_var('u', prs, level=700)
    v700  = load_var('v', prs, level=700)
    t700  = load_var('t', prs, level=700)
    rh700 = load_var('r', prs, level=700)
    # geopotential height: HRRR uses 'gh', GFS uses 'z'
    z_name = 'gh' if model == 'hrrr' else 'z'
    z500  = load_var(z_name, prs, level=500)
    # Precip rate
    with xr.open_dataset(sfc, engine='cfgrib', backend_kwargs={'filter_by_keys':{'typeOfLevel':'surface','shortName':'prate','stepType':'instant'}}) as ds:
        prate = ensure_2d(ds['prate'].load() * 3600)

    if u700 is None or v700 is None:
        return

    # DVA: placeholder compute: actual vorticity advection should be implemented
    dva = None
    if rh700 is not None:
        dva = gaussian_filter((rh700.values - rh700.values), sigma=1)

    # Plot setup
    outdir = os.path.join(plot_root,init); os.makedirs(outdir,exist_ok=True)
    extent=[LON_MIN,LON_MAX,LAT_MIN,LAT_MAX]
    def save(fig,name): fig.savefig(os.path.join(outdir,name),dpi=150); plt.close(fig)

    # Panel1
    if 1 in PLOT_PANELS:
        fig,ax=plt.subplots(1,1,figsize=(10,8),subplot_kw={'projection':ccrs.PlateCarree()})
        ax.set_extent(extent); ax.add_feature(COAST); ax.add_feature(STATES)
        cf=ax.contourf(prate.longitude,prate.latitude,prate,levels=[0.1,0.5,1,2.5,5,10,20,30,40,50],transform=ccrs.PlateCarree(),alpha=0.7)
        plt.colorbar(cf,ax=ax,label='mm/hr')
        cs=ax.contour(t700.longitude,t700.latitude,t700-273.15,levels=np.arange(-18,23),cmap='RdYlBu_r',transform=ccrs.PlateCarree())
        ax.clabel(cs,fmt='%d°C'); ax.set_title(format_title(init,fxx)+' Temp & Prate')
        save(fig,f'{model}_{init}_f{fxx:03d}_p1.png')

    # Panel2
    if 2 in PLOT_PANELS:
        fig,ax=plt.subplots(subplot_kw={'projection':ccrs.PlateCarree()})
        ax.set_extent(extent); ax.add_feature(COAST); ax.add_feature(STATES)
        if rh700 is not None:
            cf2=ax.contourf(rh700.longitude,rh700.latitude,rh700,levels=[65,75,85,100],transform=ccrs.PlateCarree(),alpha=0.6)
            plt.colorbar(cf2,ax=ax,label='RH%')
        lon2,lat2=np.meshgrid(u700.longitude.values[::10],u700.latitude.values[::10])
        ax.quiver(lon2,lat2,u700.values[::10,::10],v700.values[::10,::10],transform=ccrs.PlateCarree())
        ax.set_title('700mb RH & Wind'); save(fig,f'{model}_{init}_f{fxx:03d}_p2.png')

    # Panel3 stub qpf
    if 3 in PLOT_PANELS:
        fig,ax=plt.subplots(subplot_kw={'projection':ccrs.PlateCarree()})
        ax.set_extent(extent); ax.add_feature(COAST); ax.add_feature(STATES)
        ax.set_title('QPF Stub'); save(fig,f'{model}_{init}_f{fxx:03d}_p3.png')

    # Panel4
    if 4 in PLOT_PANELS:
        fig,ax=plt.subplots(subplot_kw={'projection':ccrs.PlateCarree()})
        ax.set_extent(extent); ax.add_feature(COAST); ax.add_feature(STATES)
        if z500 is not None:
            ax.contour(z500.longitude,z500.latitude,z500,levels=np.arange(4500,6000,60),transform=ccrs.PlateCarree())
        if dva is not None:
            cf4=ax.contourf(z500.longitude,z500.latitude,dva,levels=20,transform=ccrs.PlateCarree(),alpha=0.5)
            plt.colorbar(cf4,ax=ax,label='DVA')
        ax.set_title('500mb Height & DVA'); save(fig,f'{model}_{init}_f{fxx:03d}_p4.png')

    # Cleanup
    if model=='hrrr': os.remove(paths['sfc']); os.remove(paths['prs'])
    else: os.remove(paths)
    gc.collect()


def main():
    log('=== START ===')
    os.makedirs(data_dir,exist_ok=True); os.makedirs(plot_root,exist_ok=True)
    for model,cfg in MODEL_CONFIG.items():
        if not cfg['enabled']: continue
        lag = 1 if model=='hrrr' else 0
        inits = get_latest_inits(cfg['interval'],2,lag)
        for init in inits:
            for fxx in cfg['forecast_hours']:
                paths=download_grib(model,init,fxx)
                process_frame(model,paths,init,fxx)
    log('=== DONE ===')

if __name__=='__main__': main()
