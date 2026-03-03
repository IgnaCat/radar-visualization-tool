"""
Tests para funciones de cache keys y hashing — radar_common.py.

Qué testea este archivo:
1. grid2d_cache_key: Genera claves determinísticas para el caché de grillas 2D.
   - Mismos inputs → misma clave (determinismo).
   - Inputs diferentes → claves diferentes (no colisión).
   - Los filtros NO están en la clave (por diseño: se aplican post-cache).

2. w_operator_cache_key: Genera claves para el caché de operadores W.
   - Depende de radar, estrategia, volumen, geometría.
   - Es COMPARTIDO entre sesiones (geometría no cambia por sesión).

3. stable_hash / _hash_of / _stable: Funciones auxiliares de hashing.
   - Garantizan que el mismo objeto siempre produce el mismo hash.
   - Normalizan floats, tuplas, dicts para consistencia.

4. qc_signature: Firma de filtros QC para cache.
   - Solo filtros en AFFECTS_INTERP_FIELDS entran en la firma.

5. filters_affect_interpolation: Decide si filtros requieren re-gridding.
"""

import pytest
from app.services.radar_common import (
    stable_hash,
    _hash_of,
    _stable,
    _roundf,
    grid2d_cache_key,
    w_operator_cache_key,
    qc_signature,
    filters_affect_interpolation,
)
from app.models import RangeFilter


# ═══════════════════════════════════════════════════════════════════
# _roundf, _stable
# ═══════════════════════════════════════════════════════════════════

class TestStableHelpers:
    """Funciones auxiliares para normalización de datos antes de hashear."""

    def test_roundf_redondea_a_6_decimales(self):
        assert _roundf(3.14159265358979, 6) == 3.141593

    def test_roundf_con_entero(self):
        assert _roundf(42.0, 6) == 42.0

    def test_stable_normaliza_float(self):
        """Los floats se redondean para evitar differences de precision."""
        assert _stable(3.14159265) == _roundf(3.14159265, 6)

    def test_stable_normaliza_dict_ordenado(self):
        """Los dicts se ordenan por clave."""
        result = _stable({"b": 2, "a": 1})
        assert list(result.keys()) == ["a", "b"]

    def test_stable_normaliza_tupla_a_lista(self):
        """Las tuplas se convierten a listas (JSON no tiene tuplas)."""
        assert _stable((1, 2, 3)) == [1, 2, 3]


# ═══════════════════════════════════════════════════════════════════
# stable_hash, _hash_of
# ═══════════════════════════════════════════════════════════════════

class TestStableHash:
    """Hash estable: mismo input → mismo hash, siempre."""

    def test_determinismo(self):
        """Llamar dos veces con lo mismo da el mismo hash."""
        obj = {"field": "DBZH", "product": "PPI", "elevation": 0}
        assert stable_hash(obj) == stable_hash(obj)

    def test_orden_dict_no_importa(self):
        """El orden de claves en el dict no afecta el hash (sort_keys=True)."""
        h1 = stable_hash({"a": 1, "b": 2})
        h2 = stable_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_valores_distintos_hashes_distintos(self):
        h1 = stable_hash({"field": "DBZH"})
        h2 = stable_hash({"field": "ZDR"})
        assert h1 != h2

    def test_hash_of_determinismo(self):
        """_hash_of (BLAKE2b) es determinístico."""
        payload = {"v": 2, "file": "abc123", "prod": "PPI"}
        assert _hash_of(payload) == _hash_of(payload)

    def test_hash_of_distintos_payloads(self):
        h1 = _hash_of({"prod": "PPI"})
        h2 = _hash_of({"prod": "CAPPI"})
        assert h1 != h2


# ═══════════════════════════════════════════════════════════════════
# grid2d_cache_key
# ═══════════════════════════════════════════════════════════════════

class TestGrid2dCacheKey:
    """
    La cache key de la grilla 2D incluye:
    - file_hash, product, field, elevation, height, volume, interp, qc
    - session_id (para aislamiento entre sesiones)
    NO incluye filtros visuales (se aplican post-cache).
    """

    def _make_key(self, **overrides):
        """Helper para generar keys con defaults razonables."""
        defaults = dict(
            file_hash="abc123",
            product_upper="PPI",
            field_to_use="DBZH",
            elevation=0,
            cappi_height=None,
            volume="01",
            interp="Barnes2",
            qc_sig=(),
            session_id=None,
        )
        defaults.update(overrides)
        return grid2d_cache_key(**defaults)

    def test_determinismo(self):
        """Mismos inputs → misma clave."""
        k1 = self._make_key()
        k2 = self._make_key()
        assert k1 == k2

    def test_prefijo_g2d(self):
        """Las claves empiezan con 'g2d_' para identificarlas."""
        key = self._make_key()
        assert key.startswith("g2d_")

    def test_campo_distinto_clave_distinta(self):
        """DBZH y ZDR generan claves diferentes."""
        k1 = self._make_key(field_to_use="DBZH")
        k2 = self._make_key(field_to_use="ZDR")
        assert k1 != k2

    def test_producto_distinto_clave_distinta(self):
        k1 = self._make_key(product_upper="PPI")
        k2 = self._make_key(product_upper="CAPPI")
        assert k1 != k2

    def test_elevacion_distinta_clave_distinta(self):
        k1 = self._make_key(elevation=0)
        k2 = self._make_key(elevation=1)
        assert k1 != k2

    def test_volumen_distinto_clave_distinta(self):
        k1 = self._make_key(volume="01")
        k2 = self._make_key(volume="03")
        assert k1 != k2

    def test_archivo_distinto_clave_distinta(self):
        k1 = self._make_key(file_hash="abc123")
        k2 = self._make_key(file_hash="def456")
        assert k1 != k2

    def test_session_id_aisla(self):
        """Distintas sesiones generan claves distintas (aislamiento)."""
        k1 = self._make_key(session_id="session-A")
        k2 = self._make_key(session_id="session-B")
        assert k1 != k2

    def test_sin_session_es_distinto_a_con_session(self):
        k1 = self._make_key(session_id=None)
        k2 = self._make_key(session_id="session-A")
        assert k1 != k2

    def test_qc_sig_afecta_clave(self):
        """Filtros QC diferentes generan claves diferentes."""
        k1 = self._make_key(qc_sig=())
        k2 = self._make_key(qc_sig=(("RHOHV", 0.7, 1.0),))
        assert k1 != k2


# ═══════════════════════════════════════════════════════════════════
# w_operator_cache_key
# ═══════════════════════════════════════════════════════════════════

class TestWOperatorCacheKey:
    """
    W operator cache key depende de geometría del radar, NO de datos.
    Es compartido entre sesiones.
    """

    def _make_key(self, **overrides):
        defaults = dict(
            radar="RMA1",
            estrategia="0315",
            volumen="01",
            grid_shape=(18, 201, 201),
            grid_limits=((0, 10000), (-100000, 100000), (-100000, 100000)),
            weight_func="Barnes2",
            max_neighbors=None,
        )
        defaults.update(overrides)
        return w_operator_cache_key(**defaults)

    def test_determinismo(self):
        k1 = self._make_key()
        k2 = self._make_key()
        assert k1 == k2

    def test_prefijo_contiene_identificacion(self):
        """La clave incluye W_RMA1_0315_01_ para fácil identificación."""
        key = self._make_key()
        assert key.startswith("W_RMA1_0315_01_")

    def test_volumen_distinto_clave_distinta(self):
        k1 = self._make_key(volumen="01")
        k2 = self._make_key(volumen="03")
        assert k1 != k2

    def test_shape_distinta_clave_distinta(self):
        k1 = self._make_key(grid_shape=(18, 201, 201))
        k2 = self._make_key(grid_shape=(20, 301, 301))
        assert k1 != k2

    def test_max_neighbors_distinto(self):
        k1 = self._make_key(max_neighbors=None)
        k2 = self._make_key(max_neighbors=50)
        assert k1 != k2


# ═══════════════════════════════════════════════════════════════════
# qc_signature
# ═══════════════════════════════════════════════════════════════════

class TestQcSignature:
    """
    qc_signature genera una tupla con los filtros QC (RHOHV, etc.)
    que entran en la cache key. Filtros no-QC no se incluyen.
    """

    def test_sin_filtros(self):
        assert qc_signature([]) == ()
        assert qc_signature(None) == ()

    def test_filtro_rhohv_se_incluye(self):
        f = RangeFilter(field="RHOHV", min=0.7, max=1.0)
        sig = qc_signature([f])
        assert len(sig) == 1
        assert sig[0] == ("RHOHV", 0.7, 1.0)

    def test_filtro_dbzh_no_se_incluye(self):
        """DBZH no es QC → no va en la firma."""
        f = RangeFilter(field="DBZH", min=-10, max=60)
        sig = qc_signature([f])
        assert len(sig) == 0

    def test_determinismo(self):
        filters = [RangeFilter(field="RHOHV", min=0.7, max=1.0)]
        s1 = qc_signature(filters)
        s2 = qc_signature(filters)
        assert s1 == s2


# ═══════════════════════════════════════════════════════════════════
# filters_affect_interpolation
# ═══════════════════════════════════════════════════════════════════

class TestFiltersAffectInterpolation:
    """
    Decide si necesitamos re-gridear por culpa de los filtros.
    
    Reglas:
    - Filtro RHOHV sobre campo DBZH → True (QC cambia qué gates aportan)
    - Filtro RHOHV sobre campo RHOHV → False (es el mismo campo, no altera)
    - Filtro DBZH sobre campo DBZH → False (filtro visual, post-grid)
    - Sin filtros → False
    """

    def test_sin_filtros(self):
        assert filters_affect_interpolation([], "DBZH") is False
        assert filters_affect_interpolation(None, "DBZH") is False

    def test_rhohv_sobre_dbzh(self):
        """Filtrar RHOHV mientras muestro DBZH → SÍ afecta interpolación."""
        f = RangeFilter(field="RHOHV", min=0.7, max=1.0)
        assert filters_affect_interpolation([f], "DBZH") is True

    def test_rhohv_sobre_rhohv(self):
        """Filtrar RHOHV mientras muestro RHOHV → NO afecta (mismo campo QC)."""
        f = RangeFilter(field="RHOHV", min=0.7, max=1.0)
        assert filters_affect_interpolation([f], "RHOHV") is False

    def test_dbzh_sobre_dbzh(self):
        """Filtro visual sobre el mismo campo → NO afecta interpolación."""
        f = RangeFilter(field="DBZH", min=-10, max=60)
        assert filters_affect_interpolation([f], "DBZH") is False
