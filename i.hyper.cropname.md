## DESCRIPTION

*i.hyper.cropname* classifies a hyperspectral cube and names the
agricultural crop type of each pixel, writing a 2D class-code raster and,
optionally, a per-pixel confidence raster.

**input** is a 3D raster of bottom-of-atmosphere reflectance as a fraction in
[0, 1] — the kind of cube *i.hyper.atcorr* produces. The z axis is the
spectral axis, and the band centres are read from the cube so that only the
wavelengths between **min_wl** and **max_wl** are used.

Classification compares each pixel spectrum against a **spectral library** of
reference crop spectra. The built-in library,
`data/spectral_library_ghisaconus_approx.csv`, holds approximate mean spectra
for twelve crops; **library** replaces it with your own CSV, and **crops**
restricts the match to a subset of its entries.

### Methods

| **method** | Typical accuracy | Training | Notes |
|---|---|---|---|
| `sam` | 80-90 % | not needed | Spectral Angle Mapper; physics-based, always available |
| `svm` | 96-99 % | **required** | RBF-kernel support vector machine; the most accurate |
| `rf` | 92-95 % | **required** | Random forest; robust to noise |
| `lda` | about 85 % | **required** | Linear discriminant analysis; fast baseline |

`sam` is the default and needs no training data: it measures the angle
between the pixel spectrum and each library spectrum, assigns the closest
crop, and leaves the pixel unclassified (class 0) where even the best angle
exceeds **threshold** radians. The **-n** flag normalises spectra to unit
length first.

The other three methods are supervised and need **train**, a 2D raster of
class codes matching the library's `class_id` values. They require
`scikit-learn`; *i.hyper.cropname* fails with a clear message where it is
missing, rather than silently falling back to `sam`.

### Optimal Hyperspectral Narrowbands

With **n_ohnb** greater than 0 the module selects that many most
discriminating bands by PCA factor loading before classifying, following
Aneece & Thenkabail (2018), who reduced 242 Hyperion bands to 30 for
five-crop classification with little accuracy loss. **n_ohnb=0**, the
default, uses every band in the **min_wl**-**max_wl** window.

### Spectral library format

The library is a CSV whose first three columns are `name`, `source` and
`class_id`, followed by one `wl_<nm>` column per band:

```text
name,source,class_id,wl_447,wl_457,wl_467,...
corn,GHISACONUS_approx,1,0.0412,0.0438,0.0455,...
```

Lines beginning with `#` are comments. The library's own wavelengths need not
match the cube's; the reference spectra are resampled onto the cube's band
centres.

The built-in library covers twelve crops:

| `class_id` | Crop | Source |
|---|---|---|
| 1-5 | corn, soybean, winter_wheat, cotton, rice | GHISACONUS (Aneece & Thenkabail 2018) |
| 6-9 | alfalfa, almonds, grapes, pistachios | DESIS/PRISMA (Aneece et al. 2024) |
| 10-12 | wheat_pakistan, sugarcane_pakistan, maize_pakistan | CROPSPECPK (Raza et al. 2026) |

These are **approximate** spectra compiled from the literature, meant for
demonstration. For real work, replace them with calibrated field
measurements or the full GHISACONUS library from
[USGS](https://www.usgs.gov/programs/national-land-imaging-program/global-hyperspectral-imaging-spectral-library-agricultural).

### Outputs

| Output | Type | Content |
|---|---|---|
| **output** | 2D raster, integer | crop class code; 0 where no library spectrum matched |
| **confidence** | 2D raster, float | match confidence in [0, 1] |
| `<output>_legend.csv` | CSV | `class_id` to crop-name mapping, written beside the output |

The **-p** flag prints the band and library statistics that classification
would use — band count, the wavelength window actually matched, and which
library entries were selected — and then exits without writing any raster.
It is the quickest way to check that a cube's spectral axis and a library
agree before committing to a full run.

## NOTES

The module reads the band centres from the cube's own metadata, as
*i.hyper.import* writes them. **Where a cube carries no wavelength metadata,
the module warns and then assumes 1 nm spacing starting at min_wl.** That
assumption is almost never right for a real sensor — EMIT and PRISMA are
near 7.4 nm and 10 nm — and it silently shifts every band against the
library, so the resulting crop map can be confidently wrong. Check for
`No wavelength metadata found` in the output, and if it appears, fix the
cube's metadata rather than trusting the result. The **-p** flag reports the
cube and library wavelength ranges it would use, so it catches this before
any raster is written.

`sam` runs with nothing beyond `numpy`. `svm`, `rf` and `lda` additionally
need `scikit-learn`, imported only after the parser has run, so `--help` and
**-p** work without it.

Besides GRASS, the module runs standalone on GeoTIFF and HDF5 cubes through
[libras3d](https://github.com/YannChemin/libras3d), built with
`DEBIAN_BUILD=1`, with no GRASS installation. In that mode `RAS3D_PATH` and
`RAS3D_OUTDIR` locate the input scenes and the output directory.

## EXAMPLES

### Spectral Angle Mapper with the built-in library

```sh
i.hyper.cropname input=hyperspectral_cube@PERMANENT \
    output=crop_map confidence=crop_conf \
    method=sam threshold=0.08
```

### Check the cube against the library before classifying

```sh
i.hyper.cropname -p input=hyperspectral_cube output=dummy
```

### Supervised classification with training samples

```sh
i.hyper.cropname input=hyperspectral_cube@PERMANENT \
    output=crop_map_svm train=training_samples method=svm
```

### Restrict the library to a few irrigated crops

```sh
i.hyper.cropname input=cube output=irr_crops \
    crops=alfalfa,corn,cotton,grapes method=sam
```

### Reduce to 30 optimal narrowbands over the VNIR-SWIR window

```sh
i.hyper.cropname input=cube output=crop_map \
    n_ohnb=30 min_wl=450 max_wl=2400
```

### Standalone, outside GRASS

```sh
export RAS3D_PATH=/data/scenes
export RAS3D_OUTDIR=/data/output
i.hyper.cropname input=/data/scenes/wyvern.tiff \
    output=wyvern_crops confidence=wyvern_conf \
    method=sam threshold=0.08
```

## SEE ALSO

*[i.hyper.atcorr](i.hyper.atcorr.md),
[i.hyper.endmembers](i.hyper.endmembers.md),
[i.hyper.lib_ossl](i.hyper.lib_ossl.md),
[i.hyper.speclookup](i.hyper.speclookup.md),
[i.hyper.spectroscopy](i.hyper.spectroscopy.md),
[r3.what](r3.what.md)*

GHISACONUS spectral library:
[USGS Global Hyperspectral Imaging Spectral Library of Agricultural
Crops](https://www.usgs.gov/programs/national-land-imaging-program/global-hyperspectral-imaging-spectral-library-agricultural)

## REFERENCES

Aneece, I. & Thenkabail, P.S. (2018). *Accuracies achieved in classifying
five leading world crop types and their growth stages using optimal
Earth observing-1 Hyperion hyperspectral narrowbands on Google Earth
Engine.* Remote Sensing 10, 2027.
[doi:10.3390/rs10122027](https://doi.org/10.3390/rs10122027)

Aneece, I. & Thenkabail, P.S. (2021). *Classifying crop types using two
generations of hyperspectral sensors (Hyperion and DESIS) with machine
learning on the cloud.* Remote Sensing 13, 4704.
[doi:10.3390/rs13224704](https://doi.org/10.3390/rs13224704)

Aneece, I., Thenkabail, P.S., McCormick, R., Alifu, H., Foley, D.,
Oliphant, A.J., Teluguntla, P. (2024). *Machine learning and new-generation
spaceborne hyperspectral data advance crop type mapping.* Photogrammetric
Engineering & Remote Sensing 90(11), 687-698.

Basener, B. (2022). *Classifying crop types using Gaussian Bayesian models
and neural networks on GHISACONUS USGS data.* arXiv:2207.11228.

Raza, D. et al. (2026). *Development of a spectral library of crops
(CROPSPECPK).* BMC Plant Biology 26:561.

## AUTHORS

Yann Chemin
