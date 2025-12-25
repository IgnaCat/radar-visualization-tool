import os, sys, platform

import rasterio

print("Python:", sys.version)
print("Plataforma:", platform.platform())
print("rasterio:", rasterio.__version__)
print("GDAL via rasterio:", getattr(rasterio, "__gdal_version__", "n/a"))
print("rasterio wheel path:", rasterio.__file__)

# ¿También hay osgeo.gdal instalado? (no es necesario si usás rasterio/titiler)
try:
    from osgeo import gdal
    print("osgeo.gdal (py):", gdal.__version__, "runtime:", gdal.VersionInfo())
except Exception as e:
    print("osgeo.gdal: NO instalado o no disponible ->", repr(e))

# ¿Hay rutas de QGIS/OSGeo en el PATH que puedan mezclar DLLs?
print("PATH contiene OSGeo/QGIS?:", any("OSGeo" in p or "QGIS" in p for p in os.environ.get("PATH","").split(os.pathsep)))

