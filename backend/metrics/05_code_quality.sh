#!/usr/bin/env bash
# =============================================================================
# 05_code_quality.sh
# Genera métricas de calidad del código para:
#   - tab:cobertura       (pytest --cov por módulo)
#   - tab:mantenibilidad  (radon mi / cc / raw + LOC frontend)
#
# Uso (desde la raíz del proyecto):
#   bash backend/metrics/05_code_quality.sh
#
# Requisitos:
#   pip install pytest pytest-cov radon   (en el entorno del backend)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND="$SCRIPT_DIR/.."
ROOT="$BACKEND/.."

cd "$BACKEND"

# Detectar ejecutable Python del venv (prioridad sobre el del sistema)
if [ -x "$BACKEND/venv/Scripts/python.exe" ]; then
    PY="$BACKEND/venv/Scripts/python.exe"
elif [ -x "$BACKEND/venv/bin/python" ]; then
    PY="$BACKEND/venv/bin/python"
elif [ -n "$VIRTUAL_ENV" ] && command -v python &>/dev/null; then
    PY=python
elif command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "ERROR: No se encontró Python. Activá el venv primero." >&2
    exit 1
fi

DIVIDER="================================================================"

echo "$DIVIDER"
echo "  05_code_quality.sh — Métricas de calidad del código"
echo "$DIVIDER"

# =============================================================================
# 1. Cobertura de pruebas (pytest --cov)
# =============================================================================
echo ""
echo "$DIVIDER"
echo "  [1/3] Cobertura de pruebas (pytest --cov)"
echo "$DIVIDER"

# Generar reporte JSON para parseo por módulo + reporte terminal para visibilidad
$PY -m pytest tests/ \
    --cov=app \
    --cov-report=term-missing \
    --cov-report=json:"$SCRIPT_DIR/cache/coverage.json" \
    -q 2>&1 || true   # || true: continuar aunque haya tests fallidos

# Parsear coverage.json y mostrar por módulo
echo ""
echo "── Cobertura por módulo (para tab:cobertura) ────────────────────"
COV_JSON="$SCRIPT_DIR/cache/coverage.json" $PY - <<'PYEOF'
import json, sys, os
from pathlib import Path

json_path = Path(os.environ["COV_JSON"])
if not json_path.exists():
    print("  [ERROR] No se encontró coverage.json — omitiendo desglose por módulo.")
    sys.exit(0)

with open(json_path) as f:
    cov = json.load(f)

modules = {
    "services/radar_processing": {"stmts": 0, "miss": 0},
    "services/orchestrators":    {"stmts": 0, "miss": 0},
    "routers":                   {"stmts": 0, "miss": 0},
    "core":                      {"stmts": 0, "miss": 0},
    "OTHER":                     {"stmts": 0, "miss": 0},
}

for filepath, data in cov["files"].items():
    # Normalizar separadores
    fp = filepath.replace("\\", "/")
    # Identificar módulo
    matched = False
    for mod in list(modules.keys())[:-1]:
        if mod in fp:
            modules[mod]["stmts"] += data["summary"]["num_statements"]
            modules[mod]["miss"]  += data["summary"]["missing_lines"]
            matched = True
            break
    if not matched:
        modules["OTHER"]["stmts"] += data["summary"]["num_statements"]
        modules["OTHER"]["miss"]  += data["summary"]["missing_lines"]

total_stmts = cov["totals"]["num_statements"]
total_miss  = cov["totals"]["missing_lines"]
total_pct   = cov["totals"]["percent_covered"]

print(f"  {'Módulo':<35} {'Cobertura':>10}")
print(f"  {'-'*35} {'-'*10}")
for mod, d in modules.items():
    if d["stmts"] == 0:
        continue
    pct = (d["stmts"] - d["miss"]) / d["stmts"] * 100
    label = mod if mod != "OTHER" else "(resto)"
    print(f"  {label:<35} {pct:>9.1f}%")
print(f"  {'-'*35} {'-'*10}")
print(f"  {'TOTAL':<35} {total_pct:>9.1f}%")
print()
print("  Rows LaTeX para tab:cobertura:")
print()
for mod, d in modules.items():
    if d["stmts"] == 0:
        continue
    pct = (d["stmts"] - d["miss"]) / d["stmts"] * 100
    label = mod if mod != "OTHER" else "(resto)"
    tex_label = r"\texttt{" + label.replace("/", r"/") + r"}"
    print(f"  {tex_label} & {pct:.0f}\\% \\\\")
print(f"  \\textbf{{Total}} & {total_pct:.0f}\\% \\\\")
PYEOF


# =============================================================================
# 2. Métricas radon
# =============================================================================
echo ""
echo "$DIVIDER"
echo "  [2/3] Mantenibilidad y complejidad (radon)"
echo "$DIVIDER"

echo ""
echo "── Índice de Mantenibilidad (radon mi -s) ───────────────────────"
$PY -m radon mi app -s 2>&1 | head -80

echo ""
echo "── Resumen promedio MI ──────────────────────────────────────────"
$PY - <<'PYEOF'
import subprocess, sys
result = subprocess.run(
    [sys.executable, "-m", "radon", "mi", "app", "-s", "--json"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("  [ERROR] radon mi falló:", result.stderr[:200])
    sys.exit(0)

import json
data = json.loads(result.stdout)
scores = []
for filepath, entry in data.items():
    if isinstance(entry, dict) and "mi" in entry:
        scores.append(entry["mi"])
    elif isinstance(entry, list):
        for e in entry:
            if isinstance(e, dict) and "mi" in e:
                scores.append(e["mi"])

if scores:
    avg_mi = sum(scores) / len(scores)
    print(f"  Archivos analizados : {len(scores)}")
    print(f"  MI promedio         : {avg_mi:.1f}  (escala 0–100; >65 = mantenible)")
    print()
    print(f"  Row LaTeX para tab:mantenibilidad:")
    print(f"  Índice de mantenibilidad promedio (MI) & {avg_mi:.1f} \\\\")
else:
    print("  [AVISO] No se pudieron parsear los scores de MI.")
PYEOF

echo ""
echo "── Complejidad ciclomática (radon cc -s -a) ─────────────────────"
$PY -m radon cc app -s -a 2>&1 | tail -15

echo ""
echo "── Resumen CC ───────────────────────────────────────────────────"
$PY - <<'PYEOF'
import subprocess, sys
result = subprocess.run(
    [sys.executable, "-m", "radon", "cc", "app", "-s", "-a", "--json"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("  [ERROR] radon cc falló:", result.stderr[:200])
    sys.exit(0)

import json
try:
    data = json.loads(result.stdout)
except json.JSONDecodeError:
    print("  [AVISO] No se pudo parsear JSON de radon cc")
    sys.exit(0)

all_cc = []
for filepath, blocks in data.items():
    if isinstance(blocks, list):
        for b in blocks:
            if isinstance(b, dict) and "complexity" in b:
                all_cc.append(b["complexity"])

if all_cc:
    avg_cc = sum(all_cc) / len(all_cc)
    max_cc = max(all_cc)
    # Grado según escala radon: A=1-5, B=6-10, C=11-15, D=16-20, E=21-25, F=26+
    def cc_grade(c):
        if c <= 5:  return "A"
        if c <= 10: return "B"
        if c <= 15: return "C"
        if c <= 20: return "D"
        if c <= 25: return "E"
        return "F"
    print(f"  Funciones/métodos   : {len(all_cc)}")
    print(f"  CC promedio         : {avg_cc:.1f}  (grado {cc_grade(avg_cc)})")
    print(f"  CC máximo           : {max_cc}  (grado {cc_grade(max_cc)})")
    print()
    print(f"  Row LaTeX para tab:mantenibilidad:")
    print(f"  Complejidad ciclomática promedio & {avg_cc:.1f} (grado {cc_grade(avg_cc)}) \\\\")
else:
    print("  [AVISO] No se encontraron bloques de código en radon cc.")
PYEOF


# =============================================================================
# 3. Líneas de código (LOC)
# =============================================================================
echo ""
echo "$DIVIDER"
echo "  [3/3] Líneas de código (radon raw)"
echo "$DIVIDER"

echo ""
echo "── Backend LOC (radon raw app -s) ───────────────────────────────"
$PY -m radon raw app -s 2>&1 | tail -10

echo ""
echo "── Backend LOC (resumen) ────────────────────────────────────────"
$PY - <<'PYEOF'
import subprocess, sys
result = subprocess.run(
    [sys.executable, "-m", "radon", "raw", "app", "-s", "--json"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("  [ERROR] radon raw falló:", result.stderr[:200])
    sys.exit(0)

import json
try:
    data = json.loads(result.stdout)
except json.JSONDecodeError:
    print("  [AVISO] No se pudo parsear JSON de radon raw")
    sys.exit(0)

# El último elemento puede ser el total, o sumar manualmente
loc_total  = sum(v.get("loc",  0) for v in data.values() if isinstance(v, dict))
sloc_total = sum(v.get("sloc", 0) for v in data.values() if isinstance(v, dict))
print(f"  LOC  (líneas totales)           : {loc_total:,}")
print(f"  SLOC (líneas de código efectivo): {sloc_total:,}")
print()
print(f"  Row LaTeX para tab:mantenibilidad:")
print(f"  Líneas de código (backend)  & {loc_total:,} ({sloc_total:,} SLOC) \\\\")
PYEOF

echo ""
echo "── Frontend LOC ─────────────────────────────────────────────────"
FRONTEND="$ROOT/frontend/src"
if [ -d "$FRONTEND" ]; then
    LOC_FRONTEND=$(find "$FRONTEND" \( -name "*.jsx" -o -name "*.js" -o -name "*.css" \) | \
                   xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}')
    echo "  Frontend src/ (.jsx + .js + .css): ${LOC_FRONTEND} líneas"
    echo ""
    echo "  Row LaTeX para tab:mantenibilidad:"
    echo "  Líneas de código (frontend) & ${LOC_FRONTEND} \\\\"
else
    echo "  [AVISO] No se encontró frontend/src — omitiendo LOC frontend."
fi


# =============================================================================
# Resumen final para copiar al .tex
# =============================================================================
echo ""
echo "$DIVIDER"
echo "  RESUMEN — rows LaTeX para tab:mantenibilidad"
echo "$DIVIDER"
echo ""
echo "  (ver los valores individuales arriba — copiar los 4 rows)"
echo ""
echo "$DIVIDER"
echo "  Listo."
echo "$DIVIDER"
