"""
Tests that libras3d is functional for i.hyper.cropname standalone mode.

Run without GRASS:
    pytest testsuite/test_ras3d_cropname.py -v
"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_ras3d_common import (
    WYVERN_PATH, TANAGER_PATH,
    skip_without_ras3d, skip_without_wyvern, skip_without_tanager,
    open_cube_checked, assert_band_valid, install_ras3d_shim, make_wl_sidecar,
)

MODULE_DIR = os.path.join(os.path.dirname(__file__), '..')
LIBRARY_PATH = os.path.join(MODULE_DIR, 'data',
                             'spectral_library_ghisaconus_approx.csv')

sys.path.insert(0, MODULE_DIR)

# Import the module by file path (name contains dots and hyphens)
import importlib.util as _ilu
def _load_cropname_module():
    spec = _ilu.spec_from_file_location(
        'i_hyper_cropname',
        os.path.join(MODULE_DIR, 'i.hyper.cropname.py')
    )
    mod = _ilu.module_from_spec(spec)
    sys.modules['i_hyper_cropname'] = mod
    spec.loader.exec_module(mod)
    return mod


@skip_without_ras3d
def test_shim_installs():
    """ras3d shim provides grass.script API."""
    install_ras3d_shim()
    import grass.script as gs
    assert hasattr(gs, 'raster3d_info')
    assert hasattr(gs, 'parser')


@skip_without_ras3d
def test_library_loads():
    """Built-in spectral library CSV loads with correct structure."""
    install_ras3d_shim()
    m = _load_cropname_module()
    names, class_ids, lib_wl, lib_spectra = m.load_library(LIBRARY_PATH)
    assert len(names) >= 5, "Expected at least 5 crops"
    assert 'corn' in names
    assert 'soybean' in names
    assert lib_wl.min() >= 400
    assert lib_wl.max() <= 2500
    assert lib_spectra.shape == (len(names), len(lib_wl))
    assert (lib_spectra >= 0).all() and (lib_spectra <= 1).all()


@skip_without_ras3d
def test_library_filter():
    """load_library respects crop_filter."""
    install_ras3d_shim()
    m = _load_cropname_module()
    names, _, _, _ = m.load_library(LIBRARY_PATH, crop_filter=['corn', 'soybean'])
    assert names == ['corn', 'soybean']


@skip_without_ras3d
def test_sam_classifier_basic():
    """SAM classifier returns valid class codes and confidences."""
    install_ras3d_shim()
    m = _load_cropname_module()
    # Create fake pixel and reference spectra (10 bands)
    rng = np.random.default_rng(42)
    refs = rng.random((5, 10)).astype(np.float32)
    pixels = refs + rng.normal(0, 0.02, refs.shape).astype(np.float32)
    pixels = np.clip(pixels, 0, 1)
    classes, conf = m.classify_sam(pixels, refs, threshold=0.15)
    assert classes.shape == (5,)
    assert conf.shape == (5,)
    assert (conf >= 0).all() and (conf <= 1).all()
    # Each perturbed pixel should match its own reference
    assert (classes > 0).all(), "All perturbed pixels should be classified"


@skip_without_ras3d
def test_sam_unclassified_above_threshold():
    """SAM sets class=0 when angle exceeds threshold."""
    install_ras3d_shim()
    m = _load_cropname_module()
    ref = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    pixel = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)  # orthogonal
    classes, conf = m.classify_sam(pixel, ref, threshold=0.01)
    assert classes[0] == 0, "Orthogonal spectrum should be unclassified"


@skip_without_ras3d
def test_ohnb_selection():
    """select_ohnb returns correct count of band indices."""
    install_ras3d_shim()
    m = _load_cropname_module()
    rng = np.random.default_rng(0)
    cube = rng.random((50, 200)).astype(np.float32)  # 50 bands, 200 pixels
    selected = m.select_ohnb(cube, n=10)
    assert len(selected) == 10
    assert len(set(selected)) == 10   # unique
    assert all(0 <= i < 50 for i in selected)


@skip_without_ras3d
@skip_without_wyvern
def test_open_wyvern_geotiff():
    """Wyvern GeoTIFF opens with 23 bands."""
    import ras3d
    h, r = open_cube_checked(WYVERN_PATH)
    assert r['depths'] == 23
    ras3d.close_cube(h)


@skip_without_ras3d
@skip_without_tanager
def test_open_tanager_hdf5():
    """Tanager HDF5 opens with 426 bands."""
    import ras3d
    h, r = open_cube_checked(TANAGER_PATH)
    assert r['depths'] == 426
    ras3d.close_cube(h)


@skip_without_ras3d
@skip_without_wyvern
def test_load_wavelengths_from_sidecar():
    """load_wavelengths reads .wl.json sidecar."""
    install_ras3d_shim()
    import ras3d; m = _load_cropname_module()
    h, r = open_cube_checked(WYVERN_PATH)
    sidecar, wl_expected = make_wl_sidecar(WYVERN_PATH, r['depths'])
    ras3d.close_cube(h)
    wl = m.load_wavelengths(WYVERN_PATH)
    assert wl is not None
    assert len(wl) == r['depths']
    os.unlink(sidecar)


@skip_without_ras3d
@skip_without_wyvern
def test_sam_end_to_end_wyvern(tmp_path):
    """SAM classification runs on Wyvern cube and writes outputs."""
    install_ras3d_shim()
    import ras3d; m = _load_cropname_module()
    os.environ['RAS3D_OUTDIR'] = str(tmp_path)

    h, r = open_cube_checked(WYVERN_PATH)
    sidecar, _ = make_wl_sidecar(WYVERN_PATH, r['depths'],
                                  start_nm=400.0, step_nm=23.5)
    ras3d.close_cube(h)

    names, class_ids, lib_wl, lib_spectra = m.load_library(LIBRARY_PATH)
    cube_wl = np.array(m.load_wavelengths(WYVERN_PATH))
    wl_mask = (cube_wl >= 400) & (cube_wl <= 1000)
    band_indices = np.where(wl_mask)[0].tolist()

    lib_res, valid = m.resample_library_to_cube(lib_wl, lib_spectra,
                                                  cube_wl[band_indices])
    band_idx_valid = [band_indices[i] for i, v in enumerate(valid) if v]

    cube = m.load_cube(WYVERN_PATH, band_idx_valid)
    B, R, C = cube.shape
    pixels = cube.reshape(B, -1).T

    valid_pix = np.isfinite(pixels).all(axis=1)
    cls, conf = m.classify_sam(pixels[valid_pix], lib_res[:, valid[valid]], threshold=0.15)

    assert cls.shape[0] == valid_pix.sum()
    assert conf.shape[0] == valid_pix.sum()
    assert (conf >= 0).all() and (conf <= 1).all()
    # At least some pixels should be classified
    classified = (cls > 0).sum()
    assert classified > 0, f"Expected >0 classified pixels, got {classified}"

    os.unlink(sidecar)


@skip_without_ras3d
@skip_without_wyvern
def test_get_region_from_shim():
    """gs.raster3d_info() shim returns correct metadata for Wyvern."""
    install_ras3d_shim()
    import grass.script as gs
    info = gs.raster3d_info(WYVERN_PATH)
    assert info['depths'] == 23
    assert info['rows']   == 7825
    assert info['cols']   == 6003
