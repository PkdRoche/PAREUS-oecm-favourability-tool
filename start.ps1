$env:SSL_CERT_FILE="C:\Users\pkdro\cacert.pem"
$env:REQUESTS_CA_BUNDLE="C:\Users\pkdro\cacert.pem"
$env:CURL_CA_BUNDLE="C:\Users\pkdro\cacert.pem"
$env:GDAL_CURL_CA_BUNDLE="C:\Users\pkdro\cacert.pem"

# Override the system-wide PROJ_LIB (set by the PostgreSQL/PostGIS installer to its
# own, older proj.db) so rasterio uses its own bundled, compatible PROJ database
# instead. Without this, any CRS operation (EPSG lookup, reprojection) fails with
# "CRSError: The EPSG code is unknown. ... DATABASE.LAYOUT.VERSION.MINOR = 2 whereas
# a number >= 6 is expected." Scoped to this process only — does not touch the
# system/user env var, so PostgreSQL/PostGIS itself is unaffected.
$env:PROJ_LIB = Join-Path $PSScriptRoot "venv\Lib\site-packages\rasterio\proj_data"
$env:PROJ_DATA = $env:PROJ_LIB

streamlit run app.py
