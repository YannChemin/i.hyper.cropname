# i.hyper.cropname

> **GitHub**: <https://github.com/YannChemin/i.hyper.cropname>

Classify hyperspectral imagery and name agricultural crop types using spectral
library matching and machine learning.

Runs inside GRASS GIS (standard raster3d API) and in **standalone mode** on
GeoTIFF/HDF5 cubes via [libras3d](https://github.com/YannChemin/libras3d)
(`DEBIAN_BUILD=1`), with no GRASS installation required.

---

## Scientific background

Hyperspectral sensors provide near-continuous spectral signatures (100s–1000s
of narrow bands, 400–2500 nm) that allow discrimination of crop types by their
unique reflectance profiles driven by:

| Region | nm | Information |
|---|---|---|
| Blue–green | 450–570 | Chlorophyll pigments, green peak |
| Red | 650–680 | Chlorophyll absorption |
| Red edge | 700–750 | Vegetation vigour, LAI |
| NIR plateau | 780–900 | Leaf internal structure |
| SWIR 1 | 1150–1250 | Leaf water content |
| SWIR 2 | 1550–1750 | Cellulose, lignin |
| SWIR 3 | 2000–2400 | Starch, protein, nitrogen |

### Classification methods

| Method | Accuracy | Notes |
|---|---|---|
| **SAM** | ~ 80–90 % | Physics-based; no training data; always available |
| **SVM** | 96–99 % | Best; requires training pixels; needs scikit-learn |
| **RF** | 92–95 % | Robust; requires training; needs scikit-learn |
| **LDA** | ~ 85 % | Fast; requires training; needs scikit-learn |

References: Aneece et al. (2024) *PE&RS* 90(11); Aneece & Thenkabail (2018)
*Remote Sens.* 10, 2027; Basener (2022) arXiv:2207.11228.

### Optimal Hyperspectral Narrowbands (OHNB)

When `n_ohnb > 0`, the module automatically selects the most discriminating
bands via PCA factor loadings — the approach of Aneece & Thenkabail (2018)
that identified 30 optimal Hyperion narrowbands from 242 for five-crop
classification with minimal accuracy loss.

---

## Built-in spectral library

`data/spectral_library_ghisaconus_approx.csv` contains **approximate**
representative mean spectra for 12 crops derived from the literature:

| Crop | Source | Class |
|---|---|---|
| corn, cotton, rice, soybean, winter_wheat | GHISACONUS (Aneece & Thenkabail 2018) | 1–5 |
| alfalfa, almonds, grapes, pistachios | DESIS/PRISMA (Aneece et al. 2024) | 6–9 |
| wheat, sugarcane, maize (Pakistan) | CROPSPECPK (Raza et al. 2026) | 10–12 |

**Replace or augment** with the full GHISACONUS library from USGS:
<https://www.usgs.gov/programs/national-land-imaging-program/global-hyperspectral-imaging-spectral-library-agricultural>

The CSV format is extensible — add rows for your own crops.

---

## Quick start

### GRASS GIS mode

```sh
# Classify with SAM using built-in library
i.hyper.cropname \
    input=hyperspectral_cube@PERMANENT \
    output=crop_map \
    confidence=crop_conf \
    method=sam threshold=0.08

# SVM with training polygons
i.hyper.cropname \
    input=hyperspectral_cube@PERMANENT \
    output=crop_map_svm \
    train=training_samples \
    method=svm

# Use only irrigated crop library entries
i.hyper.cropname \
    input=cube \
    output=irr_crops \
    crops=alfalfa,corn,cotton,grapes \
    method=sam
```

### Standalone Debian mode (no GRASS)

```sh
export RAS3D_PATH=/data/scenes
export RAS3D_OUTDIR=/data/output

i.hyper.cropname \
    input=/data/scenes/wyvern.tiff \
    output=wyvern_crops \
    confidence=wyvern_conf \
    method=sam threshold=0.08

# With HDF5 Tanager scene
i.hyper.cropname \
    input=/data/tanager.h5 \
    output=tanager_crops \
    crops=corn,soybean,winter_wheat \
    method=sam
```

---

## Outputs

| Output | Format | Contents |
|---|---|---|
| `output` | 2D raster (int) | Crop type class code (0 = unclassified) |
| `confidence` | 2D raster (float) | Classification confidence 0–1 |
| `output_legend.csv` | CSV | class_id → crop_name mapping |

---

## Options reference

| Option | Default | Description |
|---|---|---|
| `input=` | — | Input hyperspectral 3D raster (BOA reflectance) |
| `output=` | — | Output crop type map |
| `confidence=` | — | Output confidence map (optional) |
| `library=` | built-in | CSV spectral library |
| `method=` | sam | sam \| svm \| rf \| lda |
| `train=` | — | Training samples raster (required for svm/rf/lda) |
| `crops=` | all | Comma-separated crop names to search |
| `n_ohnb=` | 0 | Number of OHNBs to select (0 = all bands) |
| `threshold=` | 0.10 | SAM threshold in radians |
| `min_wl=` | 400 | Minimum wavelength (nm) |
| `max_wl=` | 2500 | Maximum wavelength (nm) |
| `-p` | | Print statistics only, no classification |
| `-n` | | Normalise spectra before SAM |
| `-v` | | Verbose output |

---

## Related repositories

| Repository | Relationship |
|---|---|
| [libras3d](https://github.com/YannChemin/libras3d) | **Upstream** — GRASS API replacement for standalone mode |
| [libsixsv](https://github.com/YannChemin/libsixsv) | **Peer** — atmospheric correction library |
| [i.hyper.atcorr](https://github.com/YannChemin/i.hyper.atcorr) | **Upstream producer** — generates BOA reflectance input |
| [i.hyper.spectroscopy](https://github.com/YannChemin/i.hyper.spectroscopy) | **Peer** — spectral feature extraction |

---

## References

- Aneece, I. & Thenkabail, P.S. (2018). Accuracies achieved in classifying five
  leading world crop types. *Remote Sensing* 10, 2027.
- Aneece, I. & Thenkabail, P.S. (2021). Classifying crop types using two
  generations of hyperspectral sensors. *Remote Sensing* 13, 4704.
- Aneece, I. et al. (2024). Machine learning and new-generation spaceborne
  hyperspectral data advance crop type mapping. *PE&RS* 90(11), 687–698.
- Basener, B. (2022). Classifying crop types using Gaussian Bayesian models and
  neural networks on GHISACONUS data. arXiv:2207.11228.
- Longchamps, L. & Philpot, W. (2023). Full-season crop phenology monitoring
  using two-dimensional normalized difference pairs. *Remote Sensing* 15, 5565.
- Maina, M.M. & Shanono, N.J. (2015). Monitoring, detection and classification
  of vegetation using hyperspectral remote sensing. *J. Eng. Technol.* 10(1).
- Rahman, M. et al. (2024). BSDR: A data-efficient deep learning-based
  hyperspectral band selection algorithm. *Sensors* 24, 7771.
- Raza, D. et al. (2026). Development of a spectral library (CROPSPECPK).
  *BMC Plant Biology* 26:561.

## License

This is free and unencumbered software released into the public domain.  
See <https://unlicense.org> for the full text.
