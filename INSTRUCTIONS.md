# OECM Favourability Tool — Input Data Instructions

This document describes the input data required to run the OECM Favourability Tool. All spatial data must be provided in **EPSG:3035 (ETRS89 / LAEA Europe)**. Data in another CRS will be reprojected automatically where possible, but EPSG:3035 is strongly recommended to avoid resampling artefacts.

## 1. Study Area Boundary

The tool needs a boundary polygon that defines the region to analyse.

- **Default (no file needed):** select a country and a NUTS2 region in the app — boundaries are fetched automatically from the Eurostat GISCO service.
- **Optional local file:** a `.gpkg`, `.geojson`, or zipped shapefile (`.zip` containing `.shp` + `.shx` + `.dbf` + `.prj`), any NUTS level (0–3), or any custom administrative boundary. Use this if you need a non-NUTS2 boundary or need to work offline.

## 2. Protected Area Network (WDPA)

A vector layer of existing protected areas, following the **World Database on Protected Areas (WDPA)** schema (UNEP-WCMC & IUCN).

- **Format:** GeoPackage (`.gpkg`), zipped shapefile (`.zip`), or GeoJSON.
- **Required attributes:** an IUCN management category field (`IUCN_CAT` or `IUCN_MAX`), and ideally a designation-type field (`DESIG`) and a realm field (`REALM` or `MARINE`) to allow marine-only sites to be excluded.
- **Source:** national protected-area registries, or an extract from [Protected Planet](https://www.protectedplanet.net/) (the WDPA/WD-OECM download tool) clipped to your area of interest.

## 3. MCE Criterion Rasters (6 layers required)

Six GeoTIFF raster layers are required, all sharing the same resolution (100 m recommended) and covering the study area. Each represents one criterion in the multi-criteria evaluation.

| Layer (file role) | What it represents | Expected value range |
|---|---|---|
| **ecosystem_condition** | A composite index of ecosystem integrity / naturalness (e.g. derived from vegetation structure, fragmentation, naturalness indices) | `[0, 1]` — 0 = fully degraded, 1 = pristine |
| **regulating_es** | Capacity to deliver regulating ecosystem services (e.g. carbon storage/sequestration, water regulation, erosion control) | `[0, 1]` |
| **cultural_es** | Capacity to deliver cultural ecosystem services (e.g. recreation potential, landscape/heritage value, spiritual significance) | `[0, 1]` |
| **provisioning_es** | Intensity/capacity of provisioning ecosystem services already in use (e.g. agricultural yield, timber extraction, fisheries) | `[0, 1]` |
| **anthropogenic_pressure** | A human-pressure index (e.g. population density, built-up density, infrastructure density) | Any positive value (not pre-normalised) — e.g. inhabitants/km² |
| **landuse** | Land cover, using **Corine Land Cover (CLC)** integer codes | Integer codes `111`–`523` (standard CLC nomenclature) |

**Practical notes:**

- The four `[0, 1]` layers do not need to be pre-normalised exactly — the tool auto-detects and rescales values that are on a `[0, 100]` scale, or on an arbitrary range, at upload time. It flags (but does not silently fix) rasters with an unusual range so you can check the interpretation is correct.
- `anthropogenic_pressure` should **not** be pre-normalised to `[0, 1]` — the tool applies its own fixed-range transformation using the pressure threshold you configure, and a pre-normalised input will distort that.
- `landuse` must be the **raw integer CLC codes**, not a rescaled/normalised layer — the tool needs the actual land-cover class to apply eliminatory rules and compatibility scoring.
- Any pixel with no embedded CRS is assumed to be EPSG:3035 and flagged with a warning; any pixel with a different CRS is reprojected automatically, also flagged.
- Negative values in the four `[0, 1]` layers are treated as invalid data (not a valid low score) and excluded — check your source data if you see a warning about masked negative pixels.

## 4. Corine Land Cover (CLC) — used for two purposes

The `landuse` raster above doubles as the source for two additional analyses in the Protection Network Diagnostic module:

- **Ecosystem representativity** — coverage of six broad ecosystem types (forests, grasslands, wetlands, water bodies, semi-natural open land, agricultural areas) against the 30% KMGBF target.
- **Ecosystem Account** — a habitat-level extent/condition/services table, selectable at CLC Level 1 (5 classes), Level 2 (15 classes) or Level 3 (44 classes, full nomenclature).

No separate file is needed for these — they reuse the `landuse` raster already supplied. A free, ready-to-use source is the [Corine Land Cover dataset](https://land.copernicus.eu/pan-european/corine-land-cover) from the Copernicus Land Monitoring Service.

## 5. Optional — Externally Proposed Candidate Sites

If you already have candidate OECM sites you want evaluated against the tool's criteria (rather than only using the sites the tool delineates automatically), you can import them at any time from the **Favourability Analysis → Import & Evaluate Sites** tab:

- **Format:** a single `.shp` (with its `.shx` / `.dbf` / `.prj` / `.cpg` sidecar files), or a `.zip` archive containing them.
- **Attributes:** any attribute schema is accepted; you can optionally select a name/ID column to label the sites in the results.

## Summary Checklist

- [ ] Study area boundary (or use the built-in NUTS2 selector — no file needed)
- [ ] WDPA protected-area layer (GeoPackage / shapefile.zip / GeoJSON)
- [ ] `ecosystem_condition.tif` — `[0,1]`
- [ ] `regulating_es.tif` — `[0,1]`
- [ ] `cultural_es.tif` — `[0,1]`
- [ ] `provisioning_es.tif` — `[0,1]`
- [ ] `anthropogenic_pressure.tif` — positive values, not pre-normalised
- [ ] `landuse.tif` — raw CLC integer codes (111–523)
- [ ] *(optional)* externally proposed candidate site shapefile(s)

All six criterion rasters and the WDPA layer must cover the same study area for the analysis to run; partial coverage triggers a warning but does not block the analysis.
