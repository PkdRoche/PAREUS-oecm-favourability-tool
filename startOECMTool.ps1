<#
    Convenience launcher for running the app locally without Docker.
    Run from the project root, with the virtual environment already created
    and activated (see README.md - "Local Installation"):

        .\startOECMTool.ps1
#>

# ---------------------------------------------------------------------------
# Optional corporate SSL certificate override.
#
# Some organisation networks intercept HTTPS traffic (corporate proxy /
# SSL inspection) and require a custom CA bundle for outbound calls to
# succeed - this app makes a couple at startup (Eurostat NUTS2 boundaries,
# optionally the WDPA API). This is OPTIONAL: most users are on a normal
# network and don't need it, and the block below does nothing unless it
# finds a certificate.
#
# If you hit an SSL error loading NUTS2 boundaries, get your organisation's
# CA bundle (.pem) and either:
#   - save it as "cacert.pem" at the project root (next to this script), or
#   - set the OECM_SSL_CERT_FILE environment variable to its path yourself
#     before running this script.
# ---------------------------------------------------------------------------
$certPath = if ($env:OECM_SSL_CERT_FILE) { $env:OECM_SSL_CERT_FILE } else { Join-Path $PSScriptRoot "cacert.pem" }
if (Test-Path $certPath) {
    Write-Host "Using custom CA bundle: $certPath"
    $env:SSL_CERT_FILE = $certPath
    $env:REQUESTS_CA_BUNDLE = $certPath
    $env:CURL_CA_BUNDLE = $certPath
    $env:GDAL_CURL_CA_BUNDLE = $certPath
}

# ---------------------------------------------------------------------------
# PROJ database override.
#
# Some machines have a system-wide PROJ_LIB environment variable pointing at
# an incompatible PROJ database - commonly set by a PostgreSQL/PostGIS
# installer. When present, it breaks every CRS operation (EPSG lookup,
# reprojection) with:
#   "CRSError: The EPSG code is unknown. ... DATABASE.LAYOUT.VERSION.MINOR = 2
#   whereas a number >= 6 is expected. It comes from another PROJ installation."
#
# This forces rasterio's own bundled, compatible PROJ database instead,
# scoped to this process only (does not touch the system/user env var, so
# PostgreSQL/PostGIS itself - if you have it - is unaffected). The path is
# resolved relative to this script, so it works for any clone of this repo.
# ---------------------------------------------------------------------------
$projData = Join-Path $PSScriptRoot "venv\Lib\site-packages\rasterio\proj_data"
if (Test-Path $projData) {
    $env:PROJ_LIB = $projData
    $env:PROJ_DATA = $projData
} else {
    Write-Warning "rasterio's bundled PROJ data not found at: $projData"
    Write-Warning "Skipping the PROJ_LIB override - if you see a CRSError below, check that your virtual environment is set up (see README.md) and try again."
}

streamlit run app.py
