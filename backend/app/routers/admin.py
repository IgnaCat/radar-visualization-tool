"""
Endpoints de administración para mantenimiento del sistema.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from typing import Optional
import time
import datetime

from ..core.cache import (
    GRID2D_CACHE, 
    W_OPERATOR_CACHE,
    get_w_operator_cache_path,
    CACHE_DIR
)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/clear-cache")
def clear_cache(
    cache_type: Optional[str] = None,
    max_age_hours: Optional[float] = None
):
    """
    Limpia cache del sistema.
    
    Args:
        cache_type: Tipo de cache a limpiar ('grid2d', 'w_operator', 'all'). Default: 'all'
        max_age_hours: Solo limpiar operadores W con edad mayor a X horas. None = limpiar todo.
    
    Returns:
        Estadísticas de limpieza
    """
    if cache_type is None:
        cache_type = 'all'
    
    cache_type = cache_type.lower()
    if cache_type not in ['grid2d', 'w_operator', 'all']:
        raise HTTPException(400, "cache_type debe ser 'grid2d', 'w_operator' o 'all'")
    
    cleared = {
        "grid2d_entries": 0,
        "w_operator_ram_entries": 0,
        "w_operator_disk_files": 0,
        "disk_space_freed_mb": 0.0
    }
    
    # Limpiar GRID2D_CACHE
    if cache_type in ['grid2d', 'all']:
        size_before = len(GRID2D_CACHE)
        GRID2D_CACHE.clear()
        cleared["grid2d_entries"] = size_before
    
    # Limpiar W_OPERATOR_CACHE (RAM + Disco)
    if cache_type in ['w_operator', 'all']:
        cleared_w = _clear_w_operator_cache(max_age_hours)
        cleared["w_operator_ram_entries"] = cleared_w["ram"]
        cleared["w_operator_disk_files"] = cleared_w["disk"]
        cleared["disk_space_freed_mb"] = cleared_w["disk_mb"]
    
    return {
        "success": True,
        "cleared": cleared,
        "message": f"Cache '{cache_type}' limpiada exitosamente"
    }


def _clear_w_operator_cache(max_age_hours: Optional[float] = None) -> dict:
    """
    Limpia cache de operadores W en RAM y disco.
    
    Args:
        max_age_hours: Solo limpiar operadores más viejos que X horas. None = limpiar todo.
    
    Returns:
        Dict con estadísticas: {"ram": int, "disk": int, "disk_mb": float}
    """
    import pickle
    
    cleared = {"ram": 0, "disk": 0, "disk_mb": 0.0}
    current_time = time.time()
    cutoff_time = current_time - (max_age_hours * 3600) if max_age_hours else 0
    
    # Limpiar RAM
    if max_age_hours is None:
        # Limpiar todo
        cleared["ram"] = len(W_OPERATOR_CACHE)
        W_OPERATOR_CACHE.clear()
    else:
        # Limpiar selectivamente por edad
        keys_to_delete = []
        for cache_key, pkg in W_OPERATOR_CACHE.items():
            metadata = pkg.get("metadata", {})
            created_at = metadata.get("created_at", 0)
            if created_at < cutoff_time:
                keys_to_delete.append(cache_key)
        
        for key in keys_to_delete:
            try:
                del W_OPERATOR_CACHE[key]
                cleared["ram"] += 1
            except Exception:
                pass
    
    # Limpiar disco
    if not CACHE_DIR.exists():
        return cleared
    
    for cache_file in CACHE_DIR.glob("W_*.npz"):
        try:
            # Verificar edad del archivo
            if max_age_hours is not None:
                metadata_file = cache_file.with_suffix('.meta.pkl')
                if metadata_file.exists():
                    with open(metadata_file, 'rb') as f:
                        metadata = pickle.load(f)
                        created_at = metadata.get("created_at", 0)
                        if created_at >= cutoff_time:
                            continue  # No borrar, es reciente
            
            # Calcular tamaño antes de borrar
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            metadata_file = cache_file.with_suffix('.meta.pkl')
            if metadata_file.exists():
                size_mb += metadata_file.stat().st_size / (1024 * 1024)
            
            # Borrar archivos
            cache_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
            
            cleared["disk"] += 1
            cleared["disk_mb"] += size_mb
            
        except Exception as e:
            print(f"Error limpiando {cache_file.name}: {e}")
    
    return cleared


@router.get("/cache-stats")
def get_cache_stats():
    """
    Obtiene estadísticas del uso de cache.
    
    Returns:
        Información sobre el estado actual de las caches
    """
    import pickle
    
    # Estadísticas GRID2D_CACHE
    grid2d_count = len(GRID2D_CACHE)
    grid2d_size_mb = sum(
        GRID2D_CACHE.getsizeof(pkg) for pkg in GRID2D_CACHE.values()
    ) / (1024 * 1024)
    
    # Estadísticas W_OPERATOR_CACHE (RAM)
    w_ram_count = len(W_OPERATOR_CACHE)
    w_ram_size_mb = sum(
        W_OPERATOR_CACHE.getsizeof(pkg) for pkg in W_OPERATOR_CACHE.values()
    ) / (1024 * 1024)
    
    # Estadísticas W_OPERATOR_CACHE (Disco)
    w_disk_count = 0
    w_disk_size_mb = 0.0
    w_disk_oldest = None
    w_disk_newest = None
    
    if CACHE_DIR.exists():
        for cache_file in CACHE_DIR.glob("W_*.npz"):
            try:
                w_disk_count += 1
                size = cache_file.stat().st_size
                w_disk_size_mb += size / (1024 * 1024)
                
                # Leer metadata para timestamps
                metadata_file = cache_file.with_suffix('.meta.pkl')
                if metadata_file.exists():
                    with open(metadata_file, 'rb') as f:
                        metadata = pickle.load(f)
                        created_at = metadata.get("created_at")
                        if created_at:
                            if w_disk_oldest is None or created_at < w_disk_oldest:
                                w_disk_oldest = created_at
                            if w_disk_newest is None or created_at > w_disk_newest:
                                w_disk_newest = created_at
            except Exception:
                pass
    
    return {
        "grid2d_cache": {
            "entries": grid2d_count,
            "size_mb": round(grid2d_size_mb, 2),
            "max_size_mb": 200
        },
        "w_operator_cache_ram": {
            "entries": w_ram_count,
            "size_mb": round(w_ram_size_mb, 2),
            "max_size_mb": 300
        },
        "w_operator_cache_disk": {
            "files": w_disk_count,
            "size_mb": round(w_disk_size_mb, 2),
            "oldest_timestamp": w_disk_oldest,
            "newest_timestamp": w_disk_newest
        }
    }


@router.get("/cache-dashboard", response_class=HTMLResponse)
def cache_dashboard():
    """
    Dashboard visual de estadísticas de cache.
    Endpoint interno - se expone públicamente como /cache en main.py
    """
    stats = get_cache_stats()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cache Stats - Radar Visualization</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 1200px;
                margin: 40px auto;
                padding: 20px;
                background: #f5f5f5;
            }}
            h1 {{ color: #333; }}
            .stats-container {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .card {{
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .card h2 {{
                margin-top: 0;
                color: #2563eb;
                font-size: 18px;
            }}
            .stat-row {{
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                border-bottom: 1px solid #eee;
            }}
            .stat-row:last-child {{ border-bottom: none; }}
            .stat-label {{ color: #666; }}
            .stat-value {{ 
                font-weight: 600; 
                color: #333;
            }}
            .progress-bar {{
                width: 100%;
                height: 20px;
                background: #e5e7eb;
                border-radius: 10px;
                overflow: hidden;
                margin-top: 10px;
            }}
            .progress-fill {{
                height: 100%;
                background: linear-gradient(90deg, #3b82f6, #2563eb);
                transition: width 0.3s ease;
            }}
            .buttons {{
                display: flex;
                gap: 10px;
                margin-top: 20px;
                flex-wrap: wrap;
            }}
            button {{
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: opacity 0.2s;
            }}
            button:hover {{ opacity: 0.8; }}
            .btn-primary {{
                background: #2563eb;
                color: white;
            }}
            .btn-danger {{
                background: #dc2626;
                color: white;
            }}
            .timestamp {{
                font-size: 12px;
                color: #999;
                margin-top: 20px;
            }}
            .alert {{
                padding: 15px;
                border-radius: 6px;
                margin-top: 20px;
                display: none;
            }}
            .alert.success {{
                background: #d1fae5;
                color: #065f46;
                border: 1px solid #6ee7b7;
            }}
            .alert.error {{
                background: #fee2e2;
                color: #991b1b;
                border: 1px solid #fca5a5;
            }}
        </style>
    </head>
    <body>
        <h1>📊 Cache Statistics</h1>
        
        <div id="alert" class="alert"></div>
        
        <div class="stats-container">
            <!-- GRID2D Cache -->
            <div class="card">
                <h2>🗺️ Grid 2D Cache (RAM)</h2>
                <div class="stat-row">
                    <span class="stat-label">Entries:</span>
                    <span class="stat-value">{stats['grid2d_cache']['entries']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Size:</span>
                    <span class="stat-value">{stats['grid2d_cache']['size_mb']:.2f} MB</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Max:</span>
                    <span class="stat-value">{stats['grid2d_cache']['max_size_mb']} MB</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {(stats['grid2d_cache']['size_mb'] / stats['grid2d_cache']['max_size_mb'] * 100) if stats['grid2d_cache']['max_size_mb'] > 0 else 0:.1f}%"></div>
                </div>
            </div>
            
            <!-- W Operator RAM -->
            <div class="card">
                <h2>⚡ W Operator Cache (RAM)</h2>
                <div class="stat-row">
                    <span class="stat-label">Entries:</span>
                    <span class="stat-value">{stats['w_operator_cache_ram']['entries']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Size:</span>
                    <span class="stat-value">{stats['w_operator_cache_ram']['size_mb']:.2f} MB</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Max:</span>
                    <span class="stat-value">{stats['w_operator_cache_ram']['max_size_mb']} MB</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {(stats['w_operator_cache_ram']['size_mb'] / stats['w_operator_cache_ram']['max_size_mb'] * 100) if stats['w_operator_cache_ram']['max_size_mb'] > 0 else 0:.1f}%"></div>
                </div>
            </div>
            
            <!-- W Operator Disk -->
            <div class="card">
                <h2>💾 W Operator Cache (Disk)</h2>
                <div class="stat-row">
                    <span class="stat-label">Files:</span>
                    <span class="stat-value">{stats['w_operator_cache_disk']['files']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Size:</span>
                    <span class="stat-value">{stats['w_operator_cache_disk']['size_mb']:.2f} MB</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Location:</span>
                    <span class="stat-value" style="font-size: 11px;">app/storage/cache/</span>
                </div>
            </div>
        </div>
        
        <div class="buttons">
            <button class="btn-primary" onclick="location.reload()">🔄 Refresh</button>
            <button class="btn-danger" onclick="clearCache('grid2d')">Clear Grid2D</button>
            <button class="btn-danger" onclick="clearCache('w_operator')">Clear W Operator</button>
            <button class="btn-danger" onclick="clearCache('all')">⚠️ Clear All</button>
        </div>
        
        <div class="timestamp">
            Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        
        <script>
            function showAlert(message, type) {{
                const alert = document.getElementById('alert');
                alert.textContent = message;
                alert.className = 'alert ' + type;
                alert.style.display = 'block';
                setTimeout(() => {{
                    alert.style.display = 'none';
                }}, 5000);
            }}
            
            async function clearCache(type) {{
                const confirmMsg = type === 'all' 
                    ? 'This will clear ALL caches. Are you sure?' 
                    : `Clear ${{type}} cache?`;
                    
                if (!confirm(confirmMsg)) return;
                
                try {{
                    const response = await fetch('/admin/clear-cache?cache_type=' + type, {{
                        method: 'POST'
                    }});
                    
                    if (!response.ok) {{
                        throw new Error('HTTP ' + response.status);
                    }}
                    
                    const data = await response.json();
                    const cleared = data.cleared;
                    let msg = 'Cache cleared: ';
                    if (cleared.grid2d_entries > 0) msg += `${{cleared.grid2d_entries}} grid2d entries, `;
                    if (cleared.w_operator_ram_entries > 0) msg += `${{cleared.w_operator_ram_entries}} W RAM entries, `;
                    if (cleared.w_operator_disk_files > 0) msg += `${{cleared.w_operator_disk_files}} W disk files (${{cleared.disk_space_freed_mb.toFixed(2)}} MB)`;
                    
                    showAlert(msg, 'success');
                    setTimeout(() => location.reload(), 1500);
                }} catch (e) {{
                    showAlert('Error clearing cache: ' + e.message, 'error');
                }}
            }}
        </script>
    </body>
    </html>
    """
    
    return html_content
