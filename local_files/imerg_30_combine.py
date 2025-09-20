from tqdm import tqdm
import os
from pathlib import Path
import h5py
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import glob
year = 2013

# Optional: shapefile plotting (pyshp). If you don't want coastline overlay, set shapefile_path = None
try:
    import shapefile as shp   # pyshp
except Exception:
    shp = None

# CONFIG
files30min = sorted(glob.glob(f'/Datastorage/saptarishi.dhanuka_asp25/imerg30min/*3IMERG.{year}*.HDF5'))
files30min = files30min  # your existing list of HDF5 paths
out_dir = Path("/Datastorage/saptarishi.dhanuka_asp25/imerg30min")
plot_dir = Path("plots")
# out_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)
shapefile_path = "ne_110m_coastline.shp"  # set to None to skip shapefile overlay
compress_level = 4

def process_hdf_to_nc_and_plot(hdf_path: Path, out_dir: Path, plot_dir: Path,
                               shapefile_path: str = None, compress_level: int = 4):
    #print(f"\nProcessing: {hdf_path.name}")
    error_count = 0
    try:
        with h5py.File(hdf_path, "r") as f:
            if "/Grid" not in f:
                raise KeyError("/Grid group not found in file")
            grid = f["/Grid"]

            # detect precipitation variable
            if "precipitation" in grid:
                pkey = "precipitation"
            elif "precipitationCal" in grid:
                pkey = "precipitationCal"
            else:
                raise KeyError("Neither 'precipitation' nor 'precipitationCal' found in /Grid")

            precip = grid[pkey][:]
            #print("  raw precip shape:", precip.shape)

            # expected HDF shapes: (1, nlat, nlon) or (nlat, nlon) or sometimes (nlon, nlat)
            nlat_hdf = precip.shape[-1]
            nlon_hdf = precip.shape[-2]

            # collapse leading time-like dim if present
            if precip.ndim == 3 and precip.shape[0] == 1:
                arr = precip[0, :, :].astype(float).copy()
            else:
                arr = precip.astype(float).copy()

            #print("  arr.shape (after collapse if any):", arr.shape)

            # try to read lon/lat from HDF if present (common names)
            lon = None
            lat = None
            for name in ("lon", "longitude", "Lon", "LON"):
                if name in grid:
                    lon = grid[name][:]
                    break
            for name in ("lat", "latitude", "Lat", "LAT"):
                if name in grid:
                    lat = grid[name][:]
                    break

            # fallback to nominal IMERG 0.1° grid
            if lon is None or lat is None:
                nx = nlon_hdf
                ny = nlat_hdf
                lon = np.linspace(-179.95, 179.95, nx)
                lat = np.linspace(89.95, -89.95, ny)  # HDF often stores north->south

            lon = np.asarray(lon)
            lat = np.asarray(lat)
            #print("  lon/lat lengths:", len(lon), len(lat), "expected (nlon,nlat):", nlon_hdf, nlat_hdf)

            # Ensure arr is (nlat, nlon). If it's transposed, fix it.
            if arr.shape == (nlat_hdf, nlon_hdf):
                pass  # OK
            elif arr.shape == (nlon_hdf, nlat_hdf):
                #print("  detected transposed array -> transposing")
                arr = arr.T
            else:
                # maybe lon/lat were swapped; attempt to detect and swap lon/lat
                if arr.shape == (len(lon), len(lat)):
                    #print("  detected lon/lat were swapped; swapping lon/lat to match arr")
                    lon, lat = lat, lon
                    if arr.shape == (len(lat), len(lon)):
                        arr = arr.T

                if arr.shape != (len(lat), len(lon)):
                    raise ValueError(f"After checks, arr.shape {arr.shape} does not match (len(lat),len(lon)) = ({len(lat)},{len(lon)})")

            # xarray likes lat increasing (south->north); if lat is descending, flip arr vertically
            if lat[0] > lat[-1]:
                lat = lat[::-1]
                arr = arr[::-1, :]

            # mask negative values (IMERG uses negatives for missing)
            arr = np.where(arr < 0, np.nan, arr)

            # build DataArray
            da = xr.DataArray(
                arr,
                dims=("lat", "lon"),
                coords={"lat": lat, "lon": lon},
                name="precipitation",
                attrs={
                    "source_file": str(hdf_path),
                    "units": "mm/hr",
                }
            )

            # dataset
            ds = da.to_dataset()

            # encoding with compression
            enc = {
                "precipitation": {
                    "zlib": True,
                    "complevel": compress_level,
                    "dtype": "float32"
                }
            }

            # save NetCDF
            out_name = out_dir / (hdf_path.stem + ".nc")
            ds.to_netcdf(path=str(out_name), mode="w", format="NETCDF4", encoding=enc)
            #print(f"  saved NC -> {out_name}")

            # # --- create and save plot ---
            # fig, ax = plt.subplots(figsize=(12, 5))
            # # choose vmin/vmax robustly
            # try:
            #     vmax = float(np.nanpercentile(da.values, 99))
            # except Exception:
            #     vmax = np.nanmax(da.values)
            # vmin = 0.0

            # # pcolormesh via xarray plotting (rasterized helps keep file size manageable)
            # da.plot.pcolormesh(ax=ax, vmin=vmin, vmax=vmax, add_colorbar=True, rasterized=True)

            # # overlay coastline shapefile safely if requested and pyshp is available
            # if shapefile_path and shp is not None:
            #     try:
            #         sf = shp.Reader(shapefile_path)
            #         for shapeRec in sf.shapeRecords():
            #             s = shapeRec.shape
            #             pts = np.asarray(s.points)
            #             parts = list(s.parts) + [len(pts)]
            #             for i in range(len(s.parts)):
            #                 start = parts[i]
            #                 end = parts[i + 1]
            #                 seg = pts[start:end]
            #                 if seg.shape[0] > 1:
            #                     ax.plot(seg[:, 0], seg[:, 1], color="white", linewidth=0.8, zorder=2)
            #     except Exception as e:
            #         print("  warning: could not plot shapefile coastlines:", e)


            # ax.set_title(hdf_path.name)
            # ax.set_xlabel("Longitude")
            # ax.set_ylabel("Latitude")

            # plot_name = plot_dir / (hdf_path.stem + ".png")
            # fig.savefig(str(plot_name), dpi=200, bbox_inches="tight")
            # plt.close(fig)
            #print(f"  saved plot -> {plot_name}")

            return out_name

    except Exception as exc:
        #print(f"  ERROR processing {hdf_path.name}: {exc}")
        error_count +=1
        print(error_count, " errors till now")
        return None, None

# Run conversion + plotting for all files
saved = []
for f in tqdm(files30min):
    p = Path(f)
    nc_files = process_hdf_to_nc_and_plot(p, out_dir, plot_dir, shapefile_path=shapefile_path, compress_level=compress_level)
