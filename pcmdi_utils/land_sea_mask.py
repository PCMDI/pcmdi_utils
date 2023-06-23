import os
import time

import global_land_mask as globe
import numpy as np
import pkg_resources
import xarray as xr
import xcdat as xc

egg_path = pkg_resources.resource_filename(
    pkg_resources.Requirement.parse("pcmdi_utils"), "share/pcmdi_utils"
)


def generate_land_sea_mask(ds, tool="pcmdi", maskname="lsmask"):
    """Generates a best guess mask on any rectilinear grid

    Args:
        ds (xarray.Dataset): target grid
        tool (str, optional): which method to use. Either "pcmdi" or "global_land_mask". Defaults to "pcmdi".
        maskname (str, optional): Variable name for returning DataArray. Defaults to "lsmask".

    Returns:
        xarray.DataArray: landsea mask on target grid. 1: land, 0: water
    """
    if tool == "pcmdi":
        mask = generate_land_sea_mask__pcmdi(
            ds,
            source=None,
            data_var="sftlf",
            maskname=maskname,
            regridTool="regrid2",
            threshold_1=0.2,
            threshold_2=0.3,
            debug=False,
        )
    elif tool == "global_land_mask":
        mask = generate_land_sea_mask__global_land_mask(ds, maskname=maskname)

    return mask


def generate_land_sea_mask__pcmdi(
    target_grid,
    source=None,
    data_var="sftlf",
    maskname="lsmask",
    regridTool="regrid2",
    threshold_1=0.2,
    threshold_2=0.3,
    debug=False,
):
    """Generates a best guess mask on any rectilinear grid, using the method described in `PCMDI's report #58`_

    Args:
        target_grid (xarray.Dataset): Either a xcdat/xarray Dataset with a grid, or a xcdat grid (rectilinear grid only)
        source (_type_, optional): A xcdat/xarray Dataset that contains a DataArray of a fractional (0.0 to 1.0) land sea mask,
        where 1 means all land. Defaults to None.
        data_var (str, optional): name of DataArray for land sea fraction/mask variable in `source`. Defaults to "sftlf".
        maskname (str, optional): Variable name for returning DataArray. Defaults to "lsmask".
        regridTool (str, optional): which xcdat regridder tool to use. Defaults to "regrid2".
        threshold_1 (float, optional): Criteria 1 for detecting cells with possible increment see report for detail difference threshold. Defaults to 0.2.
        threshold_2 (float, optional): Criteria 2 for detecting cells with possible increment see report for detail water/land content threshold. Defaults to 0.3.
        debug (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        xarray.DataArray: landsea mask on target grid. 1: land, 0: water

    References:
    .. _PCMDI's report #58: http://www-pcmdi.llnl.gov/publications/pdf/58.pdf
    """

    if source is None:
        source_path = os.path.join(egg_path, "navy_land.nc")
        ds = xc.open_dataset(source_path, decode_times=False).load()
    else:
        ds = source.copy()
        if not isinstance(ds, xr.Dataset):
            raise ValueError(
                "ERROR: type of source, ",
                type(source),
                " is not acceptable. It should be <xarray.Dataset>",
            )

    # Regrid
    if target_grid.equals(ds):
        ds_regrid = ds.copy()  # testing purpose
    else:
        start_time_r = time.time()
        ds_regrid = ds.regridder.horizontal(data_var, target_grid, tool=regridTool)
        end_time_r = time.time()

        if debug:
            print(
                "Elapsed time (regridder.horizontal):",
                end_time_r - start_time_r,
                "seconds",
            )

        # Add missed information during the regrid process
        # (this might be a bug... will report it to xcdat repo later)
        ds_regrid[data_var].lat.attrs["axis"] = "Y"
        ds_regrid[data_var].lon.attrs["axis"] = "X"
        ds_regrid[data_var].lat.attrs["bounds"] = "lat_bnds"
        ds_regrid[data_var].lon.attrs["bounds"] = "lon_bnds"
        ds_regrid[data_var].lat.attrs["units"] = "degrees_north"

    # re-generate lat lon bounds (original bounds are 2d arrays where 1d array for each is expected)
    ds_regrid = (
        ds_regrid.drop_vars(["lat_bnds", "lon_bnds"])
        .bounds.add_bounds("Y")
        .bounds.add_bounds("X")
    )

    # First guess, anything greater than 50% is land to ignore rivers and lakes
    mask = xr.where(ds_regrid[data_var] > 0.5, 1, 0)

    if debug:
        ds_regrid[data_var + "_regrid"] = ds_regrid[data_var].copy()
        ds_regrid[data_var + "_firstGuess"] = mask

    # Improve
    UL, UC, UR, ML, MR, LL, LC, LR = _create_surrounds(
        ds_regrid, data_var=data_var, debug=debug
    )

    cont = True
    i = 0

    while cont:
        mask_improved = _improve(
            mask,
            ds_regrid,
            UL,
            UC,
            UR,
            ML,
            MR,
            LL,
            LC,
            LR,
            data_var=data_var,
            threshold_1=threshold_1,
            threshold_2=threshold_2,
            regridTool=regridTool,
            debug=debug,
        )

        if mask_improved.equals(mask) or i > 10:
            cont = False
        else:
            mask = mask_improved.astype("i")

        if debug:
            print("test i:", i)

        i += 1

    return mask_improved.rename(maskname)


def _create_surrounds(ds, data_var="sftlf", debug=False):
    start_time = time.time()
    data = ds[data_var].to_numpy()
    sh = list(data.shape)
    L = ds["lon"]
    bL = ds[ds.lon.attrs["bounds"]].to_numpy()

    L_isCircular = _isCircular(L)
    L_modulo = 360

    if _isCircular(L) and bL[-1][1] - bL[0][0] % L_modulo == 0:
        sh[0] = sh[0] - 2
    else:
        sh[0] = sh[0] - 2
        sh[1] = sh[1] - 2

    UL = np.ones(sh)
    UC = np.ones(sh)
    UR = np.ones(sh)
    ML = np.ones(sh)
    MR = np.ones(sh)
    LL = np.ones(sh)
    LC = np.ones(sh)
    LR = np.ones(sh)

    if L_isCircular and bL[-1][1] - bL[0][0] % L_modulo == 0:
        UC[:, :] = data[2:]
        LC[:, :] = data[:-2]
        ML[:, 1:] = data[1:-1, :-1]
        ML[:, 0] = data[1:-1, -1]
        MR[:, :-1] = data[1:-1, 1:]
        MR[:, -1] = data[1:-1, 0]
        UL[:, 1:] = data[2:, :-1]
        UL[:, 0] = data[2:, -1]
        UR[:, :-1] = data[2:, 1:]
        UR[:, -1] = data[2:, 0]
        LL[:, 1:] = data[:-2, :-1]
        LL[:, 0] = data[:-2, -1]
        LR[:, :-1] = data[:-2, 1:]
        LR[:, -1] = data[:-2, 0]
    else:
        UC[:, :] = data[2:, 1:-1]
        LC[:, :] = data[:-2, 1:-1]
        ML[:, :] = data[1:-1, :-2]
        MR[:, :] = data[1:-1, 2:]
        UL[:, :] = data[2:, :-2]
        UR[:, :] = data[2:, 2:]
        LL[:, :] = data[:-2, :-2]
        LR[:, :] = data[:-2, 2:]

    end_time = time.time()
    if debug:
        elapsed_time = end_time - start_time
        print("Elapsed time (_create_surrounds):", elapsed_time, "seconds")

    return UL, UC, UR, ML, MR, LL, LC, LR


def _isCircular(lons):
    baxis = lons[0]  # beginning of axis
    eaxis = lons[-1]  # end of axis
    deltaend = lons[-1] - lons[-2]  # delta between two end points
    eaxistest = eaxis + deltaend - baxis  # test end axis
    tol = 0.01 * deltaend
    if abs(eaxistest - 360) < tol:
        return True
    else:
        return False


def _improve(
    mask,
    ds_regrid,
    UL,
    UC,
    UR,
    ML,
    MR,
    LL,
    LC,
    LR,
    data_var="sftlf",
    threshold_1=0.2,
    threshold_2=0.3,
    regridTool="regrid2",
    debug=False,
):
    start_time = time.time()

    ds_mask_approx = _map2four(
        mask, ds_regrid, data_var=data_var, regridTool=regridTool, debug=debug
    )
    diff = ds_regrid[data_var] - ds_mask_approx[data_var]

    # Land point conversion
    c1 = np.greater(diff, threshold_1)  # xr.DataArray
    c2 = np.greater(ds_regrid[data_var], threshold_2)  # xr.DataArray
    c = np.logical_and(c1, c2)
    ds_regrid["c"] = c

    # Now figures out local maxima
    cUL, cUC, cUR, cML, cMR, cLL, cLC, cLR = _create_surrounds(ds_regrid, data_var="c")

    L = ds_regrid["lon"]
    bL = ds_regrid[ds_regrid.lon.attrs["bounds"]].to_numpy()

    L_modulo = 360
    L_isCircular = _isCircular(L)

    if L_isCircular and bL[-1][1] - bL[0][0] % L_modulo == 0:
        c = c[1:-1]  # elimnitates north and south poles
        tmp = ds_regrid[data_var].to_numpy()[1:-1]
    else:
        c = c[1:-1, 1:-1]  # elimnitates north and south poles
        tmp = ds_regrid[data_var].to_numpy()[1:-1, 1:-1]
    m = np.logical_and(c, np.greater(tmp, np.where(cUL, UL, 0.0)))
    m = np.logical_and(m, np.greater(tmp, np.where(cUC, UC, 0.0)))
    m = np.logical_and(m, np.greater(tmp, np.where(cUR, UR, 0.0)))
    m = np.logical_and(m, np.greater(tmp, np.where(cML, ML, 0.0)))
    m = np.logical_and(m, np.greater(tmp, np.where(cMR, MR, 0.0)))
    m = np.logical_and(m, np.greater(tmp, np.where(cLL, LL, 0.0)))
    m = np.logical_and(m, np.greater(tmp, np.where(cLC, LC, 0.0)))
    m = np.logical_and(m, np.greater(tmp, np.where(cLR, LR, 0.0)))
    # Ok now update the mask by setting these points to land
    mask2 = mask * 1.0
    if _isCircular(L) and bL[-1][1] - bL[0][0] % L_modulo == 0:
        mask2[1:-1] = xr.where(m, 1, mask[1:-1])
    else:
        mask2[1:-1, 1:-1] = xr.where(m, 1, mask[1:-1, 1:-1])

    # ocean point conversion
    c1 = np.less(diff, -threshold_1)
    c2 = np.less(ds_regrid[data_var], 1.0 - threshold_2)
    c = np.logical_and(c1, c2)
    ds_regrid["c"] = c

    cUL, cUC, cUR, cML, cMR, cLL, cLC, cLR = _create_surrounds(ds_regrid, data_var="c")

    if L_isCircular and bL[-1][1] - bL[0][0] % L_modulo == 0:
        c = c[1:-1]  # elimnitates north and south poles
        tmp = ds_regrid[data_var].to_numpy()[1:-1]
    else:
        c = c[1:-1, 1:-1]  # elimnitates north and south poles
        tmp = ds_regrid[data_var].to_numpy()[1:-1, 1:-1]
    # Now figures out local maxima
    m = np.logical_and(c, np.less(tmp, np.where(cUL, UL, 1.0)))
    m = np.logical_and(m, np.less(tmp, np.where(cUC, UC, 1.0)))
    m = np.logical_and(m, np.less(tmp, np.where(cUR, UR, 1.0)))
    m = np.logical_and(m, np.less(tmp, np.where(cML, ML, 1.0)))
    m = np.logical_and(m, np.less(tmp, np.where(cMR, MR, 1.0)))
    m = np.logical_and(m, np.less(tmp, np.where(cLL, LL, 1.0)))
    m = np.logical_and(m, np.less(tmp, np.where(cLC, LC, 1.0)))
    m = np.logical_and(m, np.less(tmp, np.where(cLR, LR, 1.0)))
    # Ok now update the mask by setting these points to ocean
    if L_isCircular and bL[-1][1] - bL[0][0] % L_modulo == 0:
        mask2[1:-1] = xr.where(m, 0, mask2[1:-1])
    else:
        mask2[1:-1, 1:-1] = xr.where(m, 0, mask2[1:-1, 1:-1])

    end_time = time.time()
    if debug:
        elapsed_time = end_time - start_time
        print("Elapsed time (_improve):", elapsed_time, "seconds")

    return mask2


def _map2four(mask, ds_regrid, data_var="sftlf", regridTool="regrid2", debug=False):
    start_time = time.time()

    lons = ds_regrid.lon.to_numpy()
    lats = ds_regrid.lat.to_numpy()
    lonso = lons[::2]
    lonse = lons[1::2]
    latso = lats[::2]
    latse = lats[1::2]

    ds_tmp = ds_regrid.copy()
    ds_tmp[data_var] = mask

    start_time_c = time.time()

    oo = xc.create_uniform_grid(
        latso[0],
        latso[-1],
        latso[1] - latso[0],
        lonso[0],
        lonso[-1],
        lonso[1] - lonso[0],
    )
    oe = xc.create_uniform_grid(
        latso[0],
        latso[-1],
        latso[1] - latso[0],
        lonse[0],
        lonse[-1],
        lonse[1] - lonse[0],
    )
    eo = xc.create_uniform_grid(
        latse[0],
        latse[-1],
        latse[1] - latse[0],
        lonso[0],
        lonso[-1],
        lonso[1] - lonso[0],
    )
    ee = xc.create_uniform_grid(
        latse[0],
        latse[-1],
        latse[1] - latse[0],
        lonse[0],
        lonse[-1],
        lonse[1] - lonse[0],
    )

    end_time_c = time.time()

    doo = ds_tmp.regridder.horizontal(data_var, oo, tool=regridTool)
    doe = ds_tmp.regridder.horizontal(data_var, oe, tool=regridTool)
    deo = ds_tmp.regridder.horizontal(data_var, eo, tool=regridTool)
    dee = ds_tmp.regridder.horizontal(data_var, ee, tool=regridTool)

    end_time_r = time.time()

    out = np.zeros(mask.shape, dtype="f")

    out[::2, ::2] = doo[data_var].to_numpy()
    out[::2, 1::2] = doe[data_var].to_numpy()
    out[1::2, ::2] = deo[data_var].to_numpy()
    out[1::2, 1::2] = dee[data_var].to_numpy()

    ds_out = ds_regrid.copy()
    ds_out[data_var] = (("lat", "lon"), out)

    end_time_o = time.time()

    end_time = time.time()
    if debug:
        elapsed_time = end_time - start_time
        print("Elapsed time (_map2four):", elapsed_time, "seconds")
        print(
            "Elapsed time (_map2four, create_uniform_grid):",
            end_time_c - start_time_c,
            "seconds",
        )
        print(
            "Elapsed time (_map2four, regridder.horizontal):",
            end_time_r - end_time_c,
            "seconds",
        )
        print("Elapsed time (_map2four, out):", end_time_o - end_time_r, "seconds")

    return ds_out


def generate_land_sea_mask__global_land_mask(ds, maskname="lsmask"):
    """Generate land sea mask for given grid using `global-land-mask`_ tool

    Args:
        ds (xarray.Dataset): Either a xcdat/xarray Dataset with a grid, or a xcdat grid (rectilinear grid only)
        maskname (str, optional): Variable name for returning DataArray. Defaults to "lsmask".

    Returns:
        xarray.DataArray: landsea mask on target grid. 1: land, 0: water

    References:
    .. _global-land-mask: https://pypi.org/project/global-land-mask/
    """
    for key in list(ds.coords.keys()):
        if key in ["lat", "latitude"] or ds[key].attrs["axis"] == "Y":
            lat_key = key
        elif key in ["lon", "longitude"] or ds[key].attrs["axis"] == "X":
            lon_key = key

    lat = ds[lat_key].to_numpy()
    lon = ds[lon_key].to_numpy()

    # global_land_mask only works with lon range (-180, 180) notation, thus swap needed if (0, 360)
    lon_swapped = False

    if np.max(lon) > 180:
        ds = xc.swap_lon_axis(ds, (-180, 180))
        lon = ds[lon_key].to_numpy()
        lon_swapped = True

    # Generate land sea mask (True or False)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    mask_boolean = globe.is_land(lat_grid, lon_grid)

    # Convert [True or False] to [1 or 0]
    mask_gl = np.where(mask_boolean, 1, 0)

    # convert numpy ndarray to xarray data array
    mask_xr = xr.DataArray(
        mask_gl, coords=[list(lat), list(lon)], dims=["lat", "lon"], name=maskname
    )
    ds[maskname] = mask_xr

    # Convert longigure range back to original if needed
    if lon_swapped:
        ds = xc.swap_lon_axis(ds, (0, 360))

    return ds[maskname]
