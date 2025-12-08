# Radar Visualization Backend

FastAPI backend for processing meteorological radar data and serving COG (Cloud Optimized GeoTIFF) tiles via TiTiler.

## Quick Start

### Development Mode

```bash
make run
```

Starts server with hot-reload on `http://localhost:8000` (single worker for debugging).

### Production Mode

```bash
make run-prod
```

Starts server with 4 workers for better tile serving performance.

## Architecture

- **FastAPI**: Async web framework
- **TiTiler**: Dynamic COG tile serving
- **PyART**: Radar data processing
- **Rasterio/GDAL**: Geospatial data handling

## Performance Optimizations

### Overview

This backend implements a comprehensive performance optimization strategy focused on radar data processing and real-time map tile serving. The optimizations address multiple bottlenecks identified during development: I/O operations, CPU-intensive transformations, and concurrent request handling.

### Data Flow Pipeline

1. **Upload**: NetCDF radar files (100-500 MB) stored in `storage/uploads/`
2. **Grid Processing**: PyART interpolates radar sweeps to 3D Cartesian grid
3. **Product Generation**: Extract 2D products (CAPPI/PPI/COLMAX/RHI) from 3D grid
4. **COG Creation**: Convert to Cloud Optimized GeoTIFF with tiled structure
5. **Tile Serving**: TiTiler dynamically generates 256x256 PNG tiles on-demand
6. **Caching**: LRU cache stores processed grids to avoid re-computation

### GDAL Configuration

The application configures GDAL environment variables for optimal COG tile reading:

- **512 MB GDAL_CACHEMAX**: Main cache for decompressed raster blocks and intermediate transformations
- **250 MB VSI_CACHE_SIZE**: Virtual file system cache reduces disk I/O
- **GDAL_MAX_DATASET_POOL_SIZE=450**: Keeps file handles open to prevent costly reopen operations
- **GDAL_FORCE_CACHING=YES**: Prevents GDAL from closing/reopening files between tile requests
- **GDAL_NUM_THREADS=ALL_CPUS**: Enables parallel processing for GDAL operations (warping, resampling)
- **PROJ_NETWORK=OFF**: Disables remote CRS grid downloads that cause network delays

**Impact**: These settings reduce tile generation time from ~6-7s to ~2.7s per tile by eliminating redundant I/O and maximizing CPU utilization.

### Multi-Worker Architecture

Production mode uses 4 workers to overcome Python's Global Interpreter Lock (GIL):

- **Problem**: Single worker processes tiles sequentially due to GIL, even with async I/O
- **Solution**: Each worker is an independent Python process with separate GIL
- **Result**: True parallelism for concurrent tile requests from browser
- **Performance**: ~2.5x speedup (6-7s → 2.7s per tile) with 4 concurrent requests

**Why 4 workers?** Benchmarking showed optimal balance between:

- CPU cores available (4 physical cores on Ryzen 5 2400G)
- Memory overhead (~500MB per worker for GDAL caches)
- Diminishing returns beyond 4 workers for this workload

### COG Structure Optimization

Generated COG files use optimal structure for tile serving:

#### Tiled vs Stripped

- **Before**: Strip-based GeoTIFF (sequential row access)
  - Required reading entire strips even for small tiles
  - Poor random access performance
- **After**: 512x512 pixel tiles (random access optimized)
  - TiTiler reads only needed tiles from file
  - ~4x faster for sparse tile requests

#### Overviews (Image Pyramids)

- **Levels**: [2, 4, 8, 16, 32] → Covers zoom levels 0-10
- **Purpose**: Pre-computed downsampled versions of full-resolution image
- **Benefit**: Zoom level 6 reads from 1/16th resolution overview instead of full image
- **Impact**: 10-15x faster tile generation at lower zoom levels

#### Compression Strategy

- **Tested**: DEFLATE, LZW, NONE
- **Result**: No compression (NONE) chosen
- **Reason**: DEFLATE decompression 143x slower than raw reads on older CPUs
- **Trade-off**: ~3x larger files (30MB vs 10MB) but instant decompression

#### Projection

- **Web Mercator (EPSG:3857)**: Native projection for web maps
- **Benefit**: No runtime reprojection needed for tile serving
- **Alternative**: Store in radar native projection → requires expensive reprojection per tile

### Caching Strategy

Two-tier caching system minimizes redundant processing:

#### Grid Cache (`GRID2D_CACHE`)

- **Purpose**: Store collapsed 2D grids post-PyART interpolation
- **Size**: 200 MB (LRU eviction)
- **Key**: `hash(file) + product + field + elevation/height + volume`
- **Filters**: NOT included in key (applied as post-grid masks)
- **Benefit**: Skip 20-40s PyART gridding when changing filters/colormaps

#### Grid 3D Cache (`GRID3D_CACHE`)

- **Purpose**: Store full 3D grids for vertical transects
- **Size**: 600 MB (LRU eviction)
- **Use case**: Pseudo-RHI cross-sections need Z-axis data
- **Benefit**: 30-60s saved per pseudo-RHI request

#### Why No Tile Cache?

- **Considered**: Cache rendered PNG tiles to disk
- **Rejected**: Tile combinations are enormous (zoom × x × y × field × filter × colormap)
- **Alternative**: COG structure already optimizes tile reads to ~0.3-0.7s
- **Future**: Could cache tiles for zoom 6-8 (limited combinations)

### Filter Application Architecture

**Post-Grid Masking** approach chosen for flexibility:

#### Why Post-Grid?

- **Cache efficiency**: Single cached grid serves all filter combinations
- **Filter changes**: Instant (just re-mask array) vs 40s to re-grid
- **QC fields**: Cached separately, applied as boolean masks

#### How It Works

1. Grid radar data without filters (cache this)
2. Load QC fields (RHOHV, ZDR, etc.) and grid separately
3. Apply QC filters as 2D boolean mask: `grid[rhohv < 0.7] = nodata`
4. Apply visual filters (same field): Direct masking
5. Convert masked array to COG with transparency

#### Performance Impact

- Filter change: ~0.1s (masking) vs ~40s (re-gridding)
- Multiple filters: Applied in sequence to same cached grid
- Cache hit rate: ~80% in typical usage

### Frontend-Backend Optimization

#### Multi-Radar Frame Merging

- **Challenge**: Display data from 3 radars simultaneously
- **Solution**: Group layers by timestamp (240s tolerance)
- **Backend**: Returns `ProcessResponse.results[]` array (one per radar)
- **Frontend**: Merges into unified frames, sorted by `.order` field
- **Benefit**: Single timeline controls all radars in sync

#### Tile Request Patterns

- **Browser limit**: 6 concurrent tile requests per domain
- **Optimized**: 4 backend workers match typical browser concurrency
- **Strategy**: Leaflet requests tiles in viewport priority order
- **Challenge**: Pan/zoom triggers cascade of 20-50 new tile requests
- **Mitigation**: COG overviews + fast tile generation minimize loading flicker

### Performance Benchmarks

Measured on Ryzen 5 2400G (2018), 12GB RAM, NVMe SSD:

| Operation                    | Time   | Notes                        |
| ---------------------------- | ------ | ---------------------------- |
| NetCDF upload                | ~2s    | 200MB file over localhost    |
| PyART gridding (3D)          | 20-40s | First time, cached afterward |
| 2D product collapse          | 1-3s   | CAPPI/PPI/COLMAX             |
| COG generation               | 3-5s   | Includes overview creation   |
| Tile generation (cache miss) | 2.7s   | With 4 workers               |
| Tile generation (overview)   | 0.3s   | Lower zoom levels            |
| Filter change                | <0.1s  | Post-grid masking            |
| Pseudo-RHI generation        | 5-8s   | With 3D grid cached          |

### Known Limitations

#### CPU-Bound Operations

- **Bottleneck**: Tile reprojection/warping takes 80% of tile generation time
- **GDAL/Rasterio**: Sequential operations even with multi-threading
- **Python GIL**: Partially mitigated by multi-worker but still impacts some operations
- **Hardware**: Older CPU (2018) lacks modern SIMD/AVX optimizations

#### Memory Usage

- **Per worker**: ~500MB baseline + ~200MB grid cache + ~600MB 3D cache
- **4 workers**: ~5GB total memory footprint
- **Large radars**: 1000m grid resolution limits for memory management
- **Mitigation**: LRU eviction prevents unbounded growth

#### Disk I/O

- **Temporary files**: COG generation writes 30-50MB per product
- **Cleanup**: Manual cleanup endpoint required (automatic cleanup not implemented)
- **Storage**: Can accumulate 1-2GB per session without cleanup
- **SSD required**: HDD performance would degrade tile serving significantly

### Future Optimization Opportunities

1. **Static tile pre-generation**: Generate zoom 6-8 tiles during processing (instant serving)
2. **Redis tile cache**: Share tile cache across workers
3. **Async tile generation**: Queue tile requests, batch process
4. **WebP tiles**: 30% smaller than PNG with similar quality
5. **HTTP/2**: Multiplexed requests reduce connection overhead
6. **Grid quantization**: Reduce grid precision (float32 → uint16) for smaller cache footprint

## API Endpoints

- `POST /upload`: Upload NetCDF radar files
- `POST /process`: Process radar data to COG
- `GET /cog/tiles/{z}/{x}/{y}`: Serve map tiles
- `POST /process/pseudo_rhi`: Generate vertical cross-sections
- `POST /stats/area`: Compute statistics over polygons
- `GET /health`: Health check

## Development

### Environment Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate radar-env

# Or use pip
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

- `FRONTEND_ORIGINS`: CORS allowed origins
- Storage paths for uploads/cache/tmp files

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app + GDAL config
│   ├── schemas.py           # Pydantic models
│   ├── core/
│   │   ├── config.py        # Settings
│   │   ├── constants.py     # Field mappings
│   │   └── cache.py         # LRU caching
│   ├── routers/             # API endpoints
│   ├── services/            # Business logic
│   │   ├── radar_processor.py  # COG generation
│   │   └── radar_common.py     # Utilities
│   └── utils/               # Helpers
└── makefile                 # Run commands
```

## License

See LICENSE file in repository root.
