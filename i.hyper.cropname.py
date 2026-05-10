#!/usr/bin/env python3
"""
MODULE:      i.hyper.cropname
AUTHOR:      Yann Chemin
PURPOSE:     Classify hyperspectral imagery and name agricultural crop types.

             Works inside GRASS GIS (standard raster3d API) and in standalone
             mode on GeoTIFF/HDF5 cubes via libras3d (DEBIAN_BUILD=1).

METHODS:
  SAM  — Spectral Angle Mapper: physics-based, no training needed.
          Aneece & Thenkabail (2018, 2021, 2024); Maina & Shanono (2015).
  SVM  — Support Vector Machine (RBF kernel): 96-99 % accuracy on GHISACONUS.
          Aneece et al. (2024); Basener (2022).
  RF   — Random Forest: 92-95 % accuracy; robust to noise.
          Aneece et al. (2024).
  LDA  — Linear Discriminant Analysis: fast, good baseline.
          Basener (2022): QDA Bayes 85 % on GHISACONUS.

OHNB band selection (Optimal Hyperspectral Narrowbands):
  PCA factor-loading approach: Aneece & Thenkabail (2018) identified
  30 optimal Hyperion narrowbands from 242 for 5-crop classification.

SPECTRAL LIBRARY:
  Built-in approximate library (data/spectral_library_ghisaconus_approx.csv)
  covers 12 crops from GHISACONUS (corn, cotton, rice, soybean, winter_wheat),
  DESIS/PRISMA (alfalfa, almonds, grapes, pistachios), and CROPSPECPK
  (wheat, sugarcane, maize — Pakistan context).
  Replace or augment with full GHISACONUS from:
    https://www.usgs.gov/programs/national-land-imaging-program/
    global-hyperspectral-imaging-spectral-library-agricultural

REFERENCES:
  Aneece & Thenkabail (2018) Remote Sens. 10, 2027.
  Aneece & Thenkabail (2021) Remote Sens. 13, 4704.
  Aneece et al.       (2024) Photogram. Eng. Remote Sens. 90(11), 687-698.
  Basener             (2022) arXiv:2207.11228.
  Longchamps & Philpot (2023) Remote Sens. 15, 5565.
  Maina & Shanono     (2015) J. Eng. Technol. 10(1), 16-26.
  Raza et al.         (2026) BMC Plant Biology 26:561.
  Rahman et al.       (2024) Sensors 24, 7771.
"""

from __future__ import annotations

# ── ras3d standalone detection ────────────────────────────────────────────────
import os as _os
_RAS3D = False
if not _os.environ.get('GISBASE'):
    try:
        import importlib.util as _ilu
        if _ilu.find_spec('ras3d') and _ilu.find_spec('ras3d_grass_shim'):
            from ras3d_grass_shim import install as _r3_install
            _r3_install()
            _RAS3D = True
    except Exception:
        pass
# ─────────────────────────────────────────────────────────────────────────────

# %module
# % description: Classify hyperspectral imagery and name agricultural crop types
# % keyword: imagery
# % keyword: hyperspectral
# % keyword: classification
# % keyword: crop type
# % keyword: spectral library
# % keyword: SAM
# % keyword: SVM
# % keyword: machine learning
# %end

# %option G_OPT_R3_INPUT
# % key: input
# % required: yes
# % description: Input hyperspectral 3D raster (BOA reflectance, fraction 0-1)
# % guisection: Input
# %end

# %option G_OPT_R_OUTPUT
# % key: output
# % required: yes
# % description: Output 2D raster — crop type class code (integer)
# % guisection: Output
# %end

# %option G_OPT_R_OUTPUT
# % key: confidence
# % required: no
# % description: Output 2D raster — classification confidence [0,1]
# % guisection: Output
# %end

# %option G_OPT_F_INPUT
# % key: library
# % required: no
# % description: CSV spectral library (default: built-in GHISACONUS approx.)
# % guisection: Input
# %end

# %option
# % key: method
# % type: string
# % required: no
# % options: sam,svm,rf,lda
# % answer: sam
# % description: Classification method
# % guisection: Method
# %end

# %option G_OPT_R_INPUT
# % key: train
# % required: no
# % description: 2D training-samples raster (class codes matching library; required for svm/rf/lda)
# % guisection: Input
# %end

# %option
# % key: crops
# % type: string
# % required: no
# % multiple: yes
# % description: Comma-separated crop names to use from library (default: all)
# % guisection: Input
# %end

# %option
# % key: n_ohnb
# % type: integer
# % required: no
# % answer: 0
# % description: Number of OHNBs to select via PCA (0 = use all bands)
# % guisection: Method
# %end

# %option
# % key: threshold
# % type: double
# % required: no
# % answer: 0.10
# % description: SAM similarity threshold in radians (default 0.10 ~ 5.7 deg)
# % guisection: Method
# %end

# %option
# % key: min_wl
# % type: double
# % required: no
# % answer: 400.0
# % description: Minimum wavelength to use from cube (nm)
# % guisection: Method
# %end

# %option
# % key: max_wl
# % type: double
# % required: no
# % answer: 2500.0
# % description: Maximum wavelength to use from cube (nm)
# % guisection: Method
# %end

# %flag
# % key: p
# % description: Print band/library statistics only; do not classify
# %end

# %flag
# % key: n
# % description: Normalise spectra to unit vector before SAM matching
# %end

# %flag
# % key: v
# % description: Verbose progress output
# %end

import sys
import os
import csv
import json
import math
import ctypes
import ctypes.util
import numpy as np
import grass.script as gs
from typing import Optional

# ── Library path ───────────────────────────────────────────────────────────────
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_LIBRARY = os.path.join(_MODULE_DIR, 'data',
                                'spectral_library_ghisaconus_approx.csv')

# ── libgrass_g3d ctypes for band extraction ────────────────────────────────────
_G3D_LIB = None


def _load_g3d_lib():
    global _G3D_LIB
    if _G3D_LIB is not None:
        return _G3D_LIB
    if _RAS3D:
        return None   # handled via ras3d.open_cube / get_band
    gisbase = os.environ.get('GISBASE', '')
    if not gisbase:
        return None
    lib_dir = os.path.join(gisbase, 'lib')
    try:
        import re as _re
        _vh = os.path.join(gisbase, 'include', 'grass', 'version.h')
        with open(_vh) as _f:
            _m = _re.search(r'#define\s+GRASS_HEADERS_VERSION\s+"([^"]+)"', _f.read())
        hv = (_m.group(1).encode() if _m else b'')
        gis = ctypes.CDLL(os.path.join(lib_dir, 'libgrass_gis.so'),
                          mode=ctypes.RTLD_GLOBAL)
        gis.G__no_gisinit.restype = None
        gis.G__no_gisinit.argtypes = [ctypes.c_char_p]
        gis.G__no_gisinit(hv)
        lib = ctypes.CDLL(os.path.join(lib_dir, 'libgrass_g3d.so'))
        lib.Rast3d_extract_z_slice.restype = ctypes.c_int
        lib.Rast3d_extract_z_slice.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
        ]
        _G3D_LIB = lib
        return lib
    except Exception as e:
        gs.warning(f'Cannot load libgrass_g3d: {e}')
        return None


def _extract_band(raster3d: str, z: int) -> np.ndarray | None:
    """Extract z-th band (0-based) as float32 [rows, cols] numpy array."""
    if _RAS3D:
        import ras3d as _r3
        h = _r3.open_cube(raster3d)
        arr = _r3.get_band(h, z)
        _r3.close_cube(h)
        return arr
    lib = _load_g3d_lib()
    if lib is None:
        gs.fatal('Cannot extract bands: no g3d library and not in ras3d mode')
    import tempfile
    tmp = f'_cropname_band_{os.getpid()}_{z}'
    name3d, mapset3d = (raster3d.split('@') + [''])[:2]
    ret = lib.Rast3d_extract_z_slice(
        name3d.encode(), mapset3d.encode() if mapset3d else b'',
        ctypes.c_int(z), tmp.encode())
    if ret != 0:
        gs.fatal(f'Rast3d_extract_z_slice failed: band {z} of {raster3d}')
    # Read the temp raster back
    try:
        from grass.script import array as garray
        arr = garray.array()
        arr.read(tmp)
        data = np.asarray(arr, dtype=np.float32)
        gs.run_command('g.remove', type='raster', name=tmp, flags='f', quiet=True)
        return data
    except Exception as e:
        gs.fatal(f'Cannot read temp raster {tmp}: {e}')


def load_cube(raster3d: str, band_indices: list[int],
              verbose: bool = False) -> np.ndarray:
    """
    Load selected bands into (n_bands, rows, cols) float32 array.

    In ras3d mode: reads the full cube once; in GRASS mode: extracts per band.
    """
    if _RAS3D:
        import ras3d as _r3
        h = _r3.open_cube(raster3d)
        full = _r3.read_all_bands(h)   # [all_bands, rows, cols]
        _r3.close_cube(h)
        return full[band_indices]

    n = len(band_indices)
    cube = None
    for i, z in enumerate(band_indices):
        if verbose:
            gs.percent(i, n, 5)
        arr = _extract_band(raster3d, z)
        if cube is None:
            cube = np.empty((n,) + arr.shape, dtype=np.float32)
        cube[i] = arr
    if verbose:
        gs.percent(1, 1, 1)
    return cube


# ── Wavelength metadata ────────────────────────────────────────────────────────

def load_wavelengths(raster3d: str) -> list[float] | None:
    """
    Return band-centre wavelengths (nm) for the 3D raster.

    Tries: (1) .wl.json sidecar  (2) gs.raster3d_info / r3.info history.
    Returns None if no wavelength metadata found.
    """
    # Try .wl.json sidecar (written by i.hyper.atcorr / ras3d_write)
    for sfx in ('', '.tif', '.tiff', '.h5', '.hdf5'):
        base = raster3d.removesuffix(sfx) if raster3d.endswith(sfx) else raster3d
        wlp = base + '.wl.json'
        if os.path.exists(wlp):
            with open(wlp) as f:
                wl = json.load(f)
            # Convert µm → nm if values look like µm
            return [w * 1000 if w < 10 else w for w in wl]

    # Try GRASS r3.info history (written by i.hyper.atcorr in GRASS mode)
    try:
        hist = gs.read_command('r3.info', map=raster3d, flags='h')
        import re
        wl_dict = {}
        for line in hist.splitlines():
            m = re.match(r'\s*Band\s+(\d+):\s+([\d.]+)\s*(?:nm|um)', line)
            if m:
                b, wval = int(m.group(1)), float(m.group(2))
                wl_dict[b] = wval * 1000 if wval < 10 else wval
        if wl_dict:
            return [wl_dict[k] for k in sorted(wl_dict)]
    except Exception:
        pass

    return None


# ── Spectral library ───────────────────────────────────────────────────────────

def load_library(path: str, crop_filter: list[str] | None = None
                 ) -> tuple[list[str], list[int], np.ndarray, np.ndarray]:
    """
    Load spectral library CSV.

    Returns:
        names      — list of crop names
        class_ids  — list of integer class codes
        lib_wl     — array of library wavelengths (nm)
        lib_spectra — array (n_crops, n_wl) of reflectances
    """
    if not os.path.exists(path):
        gs.fatal(f'Spectral library not found: {path}')

    names, class_ids, spectra = [], [], []
    lib_wl = None

    with open(path, newline='') as f:
        for raw in f:
            if raw.startswith('#') or not raw.strip():
                continue
            break   # first non-comment line = header
        f.seek(0)
        reader = csv.DictReader(row for row in f if not row.startswith('#'))
        for row in reader:
            name = row['name'].strip()
            if crop_filter and name not in crop_filter:
                continue
            cid = int(row.get('class_id', len(names) + 1))
            wl_cols = sorted(
                [(float(k[3:]), k) for k in row if k.startswith('wl_')],
                key=lambda x: x[0]
            )
            if lib_wl is None:
                lib_wl = np.array([w for w, _ in wl_cols], dtype=np.float32)
            spectra.append([float(row[k]) for _, k in wl_cols])
            names.append(name)
            class_ids.append(cid)

    if not names:
        gs.fatal('No crops loaded from library (check --crops filter and library path)')

    return names, class_ids, lib_wl, np.array(spectra, dtype=np.float32)


# ── Band selection (OHNB) ──────────────────────────────────────────────────────

def select_ohnb(cube: np.ndarray, n: int, verbose: bool = False) -> list[int]:
    """
    Select n Optimal Hyperspectral Narrowbands via PCA factor loadings.

    Bands with the highest loading on the first n principal components are
    selected.  Ties broken by inter-class variance.  Approach follows
    Aneece & Thenkabail (2018): 30 OHNBs from 242 Hyperion bands.

    cube: (n_bands, rows*cols) — flattened spatial dimension.
    Returns sorted list of band indices.
    """
    if verbose:
        gs.message(f'Selecting {n} OHNBs from {cube.shape[0]} bands via PCA...')
    from numpy.linalg import svd
    # Flatten and subsample for speed (max 50k pixels)
    B, npix = cube.shape
    if npix > 50000:
        idx = np.random.choice(npix, 50000, replace=False)
        X = cube[:, idx].T.astype(np.float64)
    else:
        X = cube.T.astype(np.float64)
    X -= X.mean(axis=0)
    _, _, Vt = svd(X, full_matrices=False)
    # Vt rows = PCs; Vt[i, j] = loading of band j on PC i
    # Select bands with highest absolute loading across top n_ohnb PCs
    loadings = np.abs(Vt[:min(n, Vt.shape[0])]).max(axis=0)
    selected = np.argsort(loadings)[-n:]
    return sorted(selected.tolist())


# ── Spectral resampling ────────────────────────────────────────────────────────

def resample_library_to_cube(lib_wl: np.ndarray, lib_spectra: np.ndarray,
                              cube_wl: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate library spectra to the cube wavelength grid.

    Only cube bands within [lib_wl.min(), lib_wl.max()] are used.
    Returns (n_crops, n_cube_wl_valid) and a boolean mask of valid cube bands.
    """
    resampled = np.zeros((len(lib_spectra), len(cube_wl)), dtype=np.float32)
    for i, spec in enumerate(lib_spectra):
        resampled[i] = np.interp(cube_wl, lib_wl, spec,
                                 left=np.nan, right=np.nan)
    valid = ~np.any(np.isnan(resampled), axis=0)
    return resampled[:, valid], valid


# ── Classification algorithms ──────────────────────────────────────────────────

def classify_sam(pixel_spectra: np.ndarray, ref_spectra: np.ndarray,
                 threshold: float = 0.10,
                 normalise: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Spectral Angle Mapper classification.

    pixel_spectra : (npix, n_bands)
    ref_spectra   : (n_crops, n_bands)
    threshold     : maximum SAM angle (radians) for a valid match
    Returns (class_map [npix], confidence [npix 0-1]).
    """
    if normalise:
        pn = pixel_spectra / (np.linalg.norm(pixel_spectra, axis=1, keepdims=True) + 1e-12)
        rn = ref_spectra   / (np.linalg.norm(ref_spectra,   axis=1, keepdims=True) + 1e-12)
    else:
        pn, rn = pixel_spectra, ref_spectra

    # Cosine similarity: (npix, n_crops)
    dot = pn @ rn.T                                           # (npix, n_crops)
    pnorm = np.linalg.norm(pn, axis=1, keepdims=True) + 1e-12
    rnorm = np.linalg.norm(rn, axis=1, keepdims=True).T + 1e-12
    cos_sim = dot / (pnorm * rnorm)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angles  = np.arccos(cos_sim)                              # (npix, n_crops)

    best_idx   = np.argmin(angles, axis=1)
    best_angle = angles[np.arange(len(angles)), best_idx]

    classes    = np.where(best_angle <= threshold, best_idx + 1, 0)
    confidence = np.clip(1.0 - best_angle / math.pi, 0.0, 1.0)
    return classes.astype(np.int32), confidence.astype(np.float32)


def classify_sklearn(pixel_spectra: np.ndarray,
                     train_spectra: np.ndarray,
                     train_labels: np.ndarray,
                     method: str = 'svm') -> tuple[np.ndarray, np.ndarray]:
    """
    SVM / RF / LDA classification via scikit-learn.

    Returns (class_map [npix int32], confidence [npix float32]).
    """
    try:
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        gs.fatal(f'scikit-learn is required for method={method}. '
                 'Install: sudo apt-get install python3-sklearn')

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_spectra)
    Xte = scaler.transform(pixel_spectra)

    if method == 'svm':
        clf = SVC(kernel='rbf', probability=True, C=10, gamma='scale',
                  class_weight='balanced')
    elif method == 'rf':
        clf = RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                     n_jobs=-1, class_weight='balanced',
                                     random_state=42)
    elif method == 'lda':
        clf = LinearDiscriminantAnalysis()
    else:
        gs.fatal(f'Unknown method: {method}')

    clf.fit(Xtr, train_labels)
    pred = clf.predict(Xte).astype(np.int32)

    if hasattr(clf, 'predict_proba'):
        proba = clf.predict_proba(Xte)
        conf  = proba.max(axis=1).astype(np.float32)
    else:
        conf = np.ones(len(pred), dtype=np.float32)

    return pred, conf


# ── Training sample extraction ─────────────────────────────────────────────────

def extract_training(raster3d: str, train_raster: str,
                     band_indices: list[int],
                     cube_wl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract training spectra from the cube at non-zero pixels of train_raster.

    Returns (spectra [n_samples, n_bands], labels [n_samples]).
    """
    gs.message('Extracting training samples...')
    # Load training label raster
    label_band = None
    if _RAS3D:
        import ras3d_write as _rw
        from osgeo import gdal
        ds = gdal.Open(train_raster if '.' in train_raster else train_raster + '.tif')
        if ds is None:
            gs.fatal(f'Cannot open training raster: {train_raster}')
        label_band = ds.ReadAsArray().astype(np.int32).ravel()
        ds = None
    else:
        try:
            from grass.script import array as garray
            arr = garray.array()
            arr.read(train_raster)
            label_band = np.asarray(arr, dtype=np.int32).ravel()
        except Exception as e:
            gs.fatal(f'Cannot read training raster: {e}')

    # Load the cube for selected bands
    cube = load_cube(raster3d, band_indices)
    rows, cols = cube.shape[1], cube.shape[2]
    cube_flat  = cube.reshape(len(band_indices), -1).T  # (npix, bands)

    mask    = (label_band != 0) & np.isfinite(cube_flat).all(axis=1)
    spectra = cube_flat[mask]
    labels  = label_band[mask]
    gs.message(f'Training samples: {mask.sum()} pixels, '
               f'{len(np.unique(labels))} classes')
    return spectra, labels


# ── Output writing ─────────────────────────────────────────────────────────────

def write_output(class_map: np.ndarray, confidence: np.ndarray,
                 raster3d: str, output_name: str, conf_name: str | None,
                 names: list[str], class_ids: list[int]):
    """Write classification and confidence maps."""
    rows = class_map.shape[0]
    cols = class_map.shape[1]

    if _RAS3D:
        import ras3d as _r3, ras3d_write as _rw
        h = _r3.open_cube(raster3d)
        out_path = _rw.outpath(output_name)
        _rw.write_raster2d(out_path, class_map.astype(np.float32), h)
        gs.message(f'Crop type map written: {out_path}')
        if conf_name:
            _rw.write_raster2d(_rw.outpath(conf_name), confidence, h)
            gs.message(f'Confidence map written: {_rw.outpath(conf_name)}')
        _r3.close_cube(h)
    else:
        # Write via GRASS garray
        try:
            from grass.script import array as garray
            out = garray.array()
            out[...] = class_map
            out.write(output_name, overwrite=True)
            if conf_name:
                cmap = garray.array()
                cmap[...] = confidence
                cmap.write(conf_name, overwrite=True)
        except Exception as e:
            gs.fatal(f'Cannot write output rasters: {e}')

    # Write legend CSV
    leg_path = output_name + '_legend.csv'
    if _os.environ.get('RAS3D_OUTDIR'):
        leg_path = os.path.join(_os.environ['RAS3D_OUTDIR'],
                                os.path.basename(output_name) + '_legend.csv')
    with open(leg_path, 'w') as f:
        f.write('class_id,crop_name\n')
        f.write('0,unclassified\n')
        for cid, name in zip(class_ids, names):
            f.write(f'{cid},{name}\n')
    gs.message(f'Legend written: {leg_path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main(options: dict, flags: dict) -> int:
    raster3d   = options['input']
    output     = options['output']
    conf_out   = options.get('confidence') or ''
    lib_path   = options.get('library') or _DEFAULT_LIBRARY
    method     = options.get('method', 'sam').lower()
    train_name = options.get('train') or ''
    crops_str  = options.get('crops') or ''
    n_ohnb     = int(options.get('n_ohnb', 0))
    threshold  = float(options.get('threshold', 0.10))
    min_wl     = float(options.get('min_wl', 400.0))
    max_wl     = float(options.get('max_wl', 2500.0))
    flag_p     = flags.get('p', False)
    flag_n     = flags.get('n', False)
    flag_v     = flags.get('v', False)

    crop_filter = [c.strip() for c in crops_str.split(',') if c.strip()] or None

    # ── Load spectral library ─────────────────────────────────────────────────
    gs.message(f'Loading spectral library: {lib_path}')
    names, class_ids, lib_wl, lib_spectra = load_library(lib_path, crop_filter)
    gs.message(f'Loaded {len(names)} crops: {", ".join(names)}')

    # ── Get cube metadata ─────────────────────────────────────────────────────
    if _RAS3D:
        import ras3d as _r3
        h = _r3.open_cube(raster3d)
        r = _r3.get_region(h)
        _r3.close_cube(h)
        rows, cols, n_bands = r['rows'], r['cols'], r['depths']
    else:
        info = gs.raster3d_info(raster3d)
        rows   = int(info['rows'])
        cols   = int(info['cols'])
        n_bands = int(info['depths'])

    # ── Load wavelengths ──────────────────────────────────────────────────────
    cube_wl = load_wavelengths(raster3d)
    if cube_wl is None:
        gs.warning('No wavelength metadata found. Assuming 1 nm spacing from min_wl.')
        cube_wl = [min_wl + i for i in range(n_bands)]
    cube_wl = np.array(cube_wl, dtype=np.float32)

    # ── Filter wavelength range ───────────────────────────────────────────────
    wl_mask = (cube_wl >= min_wl) & (cube_wl <= max_wl)
    band_indices = np.where(wl_mask)[0].tolist()
    cube_wl_use  = cube_wl[band_indices]
    gs.message(f'Using {len(band_indices)} bands in [{min_wl:.0f}, {max_wl:.0f}] nm')

    # ── Print statistics and exit if -p flag ──────────────────────────────────
    if flag_p:
        gs.message('Spectral library wavelength range: '
                   f'{lib_wl.min():.1f} – {lib_wl.max():.1f} nm')
        gs.message(f'Cube wavelength range: {cube_wl_use.min():.1f} – '
                   f'{cube_wl_use.max():.1f} nm')
        gs.message('Crops and class IDs:')
        for n, cid in zip(names, class_ids):
            gs.message(f'  {cid:3d}  {n}')
        return 0

    # ── Resample library to cube wavelengths ──────────────────────────────────
    lib_resampled, valid_mask = resample_library_to_cube(
        lib_wl, lib_spectra, cube_wl_use)
    band_indices_valid = [band_indices[i] for i, v in enumerate(valid_mask) if v]
    cube_wl_valid      = cube_wl_use[valid_mask]
    gs.message(f'Library resampled to {int(valid_mask.sum())} overlapping bands')

    if int(valid_mask.sum()) < 5:
        gs.fatal('Too few overlapping bands between library and cube. '
                 'Check wavelength coverage and --min_wl/--max_wl range.')

    # ── OHNB band selection (optional) ────────────────────────────────────────
    if n_ohnb > 0 and n_ohnb < len(band_indices_valid):
        gs.message(f'Running OHNB selection (n={n_ohnb})...')
        # Need to load cube first for PCA
        sub_cube = load_cube(raster3d, band_indices_valid, verbose=flag_v)
        B = sub_cube.shape[0]
        flat = sub_cube.reshape(B, -1)
        # Exclude NaN pixels
        ok = np.isfinite(flat).all(axis=0)
        ohnb_local = select_ohnb(flat[:, ok], n_ohnb, verbose=flag_v)
        band_indices_valid = [band_indices_valid[i] for i in ohnb_local]
        lib_resampled      = lib_resampled[:, ohnb_local]
        cube_wl_valid      = cube_wl_valid[ohnb_local]
        gs.message(f'OHNBs selected at: '
                   f'{", ".join(f"{w:.0f}" for w in cube_wl_valid)} nm')

    # ── Load cube data ────────────────────────────────────────────────────────
    gs.message('Loading cube bands...')
    cube = load_cube(raster3d, band_indices_valid, verbose=flag_v)
    B, R, C = cube.shape
    pixels  = cube.reshape(B, -1).T.astype(np.float32)   # (npix, B)

    # Mask invalid pixels (NaN or negative)
    valid_pix = np.isfinite(pixels).all(axis=1) & (pixels >= 0).all(axis=1)

    # ── Classify ──────────────────────────────────────────────────────────────
    class_flat = np.zeros(R * C, dtype=np.int32)
    conf_flat  = np.zeros(R * C, dtype=np.float32)

    if method == 'sam':
        gs.message(f'SAM classification (threshold={threshold:.3f} rad)...')
        cls, conf = classify_sam(pixels[valid_pix], lib_resampled,
                                 threshold=threshold, normalise=flag_n)
        # Map local indices → class_ids
        cls_mapped = np.where(cls > 0,
                              np.array([0] + class_ids, dtype=np.int32)[cls],
                              0)
        class_flat[valid_pix] = cls_mapped
        conf_flat[valid_pix]  = conf

    else:
        if not train_name:
            gs.fatal(f'--train is required for method={method}')
        gs.message(f'Extracting training data for {method.upper()}...')
        tr_spectra, tr_labels = extract_training(
            raster3d, train_name, band_indices_valid, cube_wl_valid)
        gs.message(f'{method.upper()} classification...')
        cls, conf = classify_sklearn(pixels[valid_pix], tr_spectra, tr_labels,
                                     method=method)
        class_flat[valid_pix] = cls
        conf_flat[valid_pix]  = conf

    class_map  = class_flat.reshape(R, C)
    confidence = conf_flat.reshape(R, C)

    # ── Crop type summary ─────────────────────────────────────────────────────
    gs.message('Classification summary:')
    total_valid = valid_pix.sum()
    for cid, name in zip([0] + class_ids, ['unclassified'] + names):
        count = (class_map == cid).sum()
        pct   = 100.0 * count / (R * C)
        gs.message(f'  {cid:3d}  {name:<22s}  {count:8d} px  ({pct:.1f} %)')

    # ── Write outputs ─────────────────────────────────────────────────────────
    write_output(class_map, confidence, raster3d, output,
                 conf_out or None, names, class_ids)

    gs.message('i.hyper.cropname complete.')
    return 0


if __name__ == '__main__':
    options, flags = gs.parser()
    sys.exit(main(options, flags))
