from flask import Blueprint, request, jsonify
from services import radar_processor
from utils import helpers
import os

bp = Blueprint("process", __name__)

@bp.route("/process", methods=["POST"])
def process_file():
    """
    Endpoint para procesar archivos de radar.
    ---
    post:
      summary: Procesa archivos NetCDF previamente subido
      consumes:
        - application/json
      parameters:
        - in: body
          name: body
          required: True
          schema:
            type: object
            properties:
              filepaths:
                type: array
                items:
                  type: string
                example: ["/uploads/RMA1.nc", "/uploads/RMA2.nc"]
            required: [filepaths]
        responses:
        200:
          description: Archivos procesados exitosamente
          schema:
            type: object
            properties:
              animation: { type: boolean, example: True }
              outputs:
                type: array
                items:
                  type: object
                  properties:
                    image_url: { type: string }
                    metadata: { type: object }
                    bounds: { type: string }
                    field_used: { type: string }
                    source_file: { type: string }
                    timestamp: { type: string }
        400:
          description: Parámetros inválidos
        404:
          description: Archivo no encontrado
        500:
          description: Error interno
    """
    
    data = request.json
    filepaths = data.get("filepaths")

    if not filepaths or not isinstance(filepaths, list):
        return jsonify({"error": "Debe proporcionar una lista de 'filepaths'"}), 400

    try:
        # Limpiar archivos viejos
        helpers.cleanup_tmp()

        processed = []
        for path in filepaths:
            if not os.path.exists(path):
                return jsonify({"error": f"Archivo no encontrado: {path}"}), 404
            
            # Extraer metadata del nombre del archivo
            _ , _ , _ , timestamp = helpers.extract_metadata_from_filename(path)

            result = radar_processor.process_radar(path)
            result["timestamp"] = timestamp
            processed.append(result)

        if helpers.should_animate(processed):
            gif_url = helpers.create_animation([r["image_url"] for r in processed])
            return jsonify({
                "animation": True,
                "outputs": [{
                    "image_url": gif_url,
                    "bounds": processed[0].get("bounds"),
                    "field_used": processed[0].get("field_used"),
                    "timestamp": f"{processed[0].get('timestamp')} to {processed[-1].get('timestamp')}",
                    "metadata": "GIF generado a partir de archivos con mismo radar y tiempo cercano"
                }]
            }), 200

        return jsonify({
            "animation": False,
            "outputs": processed
        }), 200

    except Exception as e:
        return jsonify({"error": f"Error procesando archivos: {str(e)}"}), 500
