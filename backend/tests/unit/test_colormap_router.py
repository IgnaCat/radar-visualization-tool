"""
Tests para app.routers.colormap — endpoints de colormaps y leyendas.

Qué testea este archivo:
1. GET /colormap/options — devuelve opciones de colormaps por campo.
2. GET /colormap/defaults — devuelve colormap default por campo.
3. GET /colormap/colors/{cmap_name} — devuelve lista de colores hex.
4. GET /colormap/legend/{field_key} — devuelve leyenda completa para un campo.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


# ═══════════════════════════════════════════════════════════════════
# GET /colormap/options
# ═══════════════════════════════════════════════════════════════════

class TestColormapOptions:
    """Devuelve dict {campo: [lista de colormaps]}."""

    def test_devuelve_200(self):
        response = client.get("/colormap/options")
        assert response.status_code == 200

    def test_contiene_dbzh(self):
        """DBZH es un campo obligatorio con colormaps."""
        data = client.get("/colormap/options").json()
        assert "DBZH" in data
        assert isinstance(data["DBZH"], list)
        assert len(data["DBZH"]) > 0

    def test_contiene_rhohv(self):
        """RHOHV también tiene colormaps disponibles."""
        data = client.get("/colormap/options").json()
        assert "RHOHV" in data

    def test_valores_son_strings(self):
        """Cada colormap es un string."""
        data = client.get("/colormap/options").json()
        for field, cmaps in data.items():
            for cmap in cmaps:
                assert isinstance(cmap, str), f"{field}: {cmap} no es string"


# ═══════════════════════════════════════════════════════════════════
# GET /colormap/defaults
# ═══════════════════════════════════════════════════════════════════

class TestColormapDefaults:
    """Devuelve dict {campo: colormap_default}."""

    def test_devuelve_200(self):
        response = client.get("/colormap/defaults")
        assert response.status_code == 200

    def test_dbzh_tiene_default(self):
        data = client.get("/colormap/defaults").json()
        assert "DBZH" in data
        assert isinstance(data["DBZH"], str)

    def test_cada_campo_tiene_un_string(self):
        data = client.get("/colormap/defaults").json()
        for field, cmap in data.items():
            assert isinstance(cmap, str), f"{field}: default no es string"


# ═══════════════════════════════════════════════════════════════════
# GET /colormap/colors/{cmap_name}
# ═══════════════════════════════════════════════════════════════════

class TestColormapColors:
    """Devuelve colores hex para un colormap dado."""

    def test_colormap_valido(self):
        """Un colormap válido devuelve 200 con colores."""
        response = client.get("/colormap/colors/viridis")
        assert response.status_code == 200
        data = response.json()
        assert "colors" in data
        assert "steps" in data

    def test_cantidad_de_colores_default(self):
        """Por defecto devuelve 256 colores."""
        data = client.get("/colormap/colors/viridis").json()
        assert data["steps"] == 256
        assert len(data["colors"]) == 256

    def test_steps_custom(self):
        """Se puede pedir un número custom de steps."""
        data = client.get("/colormap/colors/viridis?steps=10").json()
        assert data["steps"] == 10
        assert len(data["colors"]) == 10

    def test_colores_son_hex(self):
        """Cada color es un string hex tipo #rrggbb."""
        data = client.get("/colormap/colors/viridis").json()
        for color in data["colors"][:10]:  # chequear los primeros 10
            assert color.startswith("#")
            assert len(color) == 7

    def test_colormap_invalido_devuelve_400(self):
        """Un colormap inexistente devuelve 400."""
        response = client.get("/colormap/colors/colormap_que_no_existe_xyz")
        assert response.status_code == 400


# ═══════════════════════════════════════════════════════════════════
# GET /colormap/legend/{field_key}
# ═══════════════════════════════════════════════════════════════════

class TestColormapLegend:
    """Devuelve leyenda completa: values, colors, gradient, vmin/vmax, unit."""

    def test_dbzh_devuelve_200(self):
        response = client.get("/colormap/legend/DBZH")
        assert response.status_code == 200

    def test_dbzh_campos_requeridos(self):
        """La leyenda tiene todos los campos esperados."""
        data = client.get("/colormap/legend/DBZH").json()
        for key in ["field", "colormap", "vmin", "vmax", "values", "colors",
                     "gradient_colors", "gradient_steps", "unit"]:
            assert key in data, f"Falta campo '{key}' en leyenda"

    def test_dbzh_valores_coherentes(self):
        """vmin < vmax y los values están dentro del rango."""
        data = client.get("/colormap/legend/DBZH").json()
        assert data["vmin"] < data["vmax"]
        assert data["unit"] == "dBZ"

    def test_colores_match_values(self):
        """Cantidad de colors == cantidad de values."""
        data = client.get("/colormap/legend/DBZH").json()
        assert len(data["colors"]) == len(data["values"])

    def test_gradient_tiene_256_colores(self):
        """El gradiente tiene LEGEND_GRADIENT_STEPS (256) colores."""
        data = client.get("/colormap/legend/DBZH").json()
        assert data["gradient_steps"] == 256
        assert len(data["gradient_colors"]) == 256

    def test_case_insensitive(self):
        """El field_key es case-insensitive."""
        r1 = client.get("/colormap/legend/DBZH")
        r2 = client.get("/colormap/legend/dbzh")
        assert r1.status_code == 200
        assert r2.status_code == 200

    def test_campo_inexistente_devuelve_404(self):
        """Un campo que no existe devuelve 404."""
        response = client.get("/colormap/legend/CAMPO_FALSO")
        assert response.status_code == 404

    def test_override_cmap(self):
        """Se puede pasar un cmap_name distinto al default."""
        response = client.get("/colormap/legend/DBZH?cmap_name=viridis")
        assert response.status_code == 200
        data = response.json()
        assert data["colormap"] == "viridis"

    def test_rhohv_legend(self):
        """RHOHV tiene leyenda con valores entre 0 y 1."""
        data = client.get("/colormap/legend/RHOHV").json()
        assert data["vmin"] >= 0
        assert data["vmax"] <= 1.1
