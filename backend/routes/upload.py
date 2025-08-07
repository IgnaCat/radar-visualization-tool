from flask import Blueprint, request, jsonify, current_app
import os
from werkzeug.utils import secure_filename
from uuid import uuid4

bp = Blueprint("upload", __name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@bp.route("/upload", methods=["POST"])
def upload_file():
    """
    Endpoint para subir múltiples archivos NetCDF.
    ---
    summary: Sube uno o más archivos NetCDF
    consumes:
      - multipart/form-data
    parameters:
      - name: files
        in: formData
        type: array
        items:
          type: file
        required: true
        description: Lista de archivos NetCDF a subir
    responses:
      200:
        description: Archivos subidos correctamente
        schema:
          type: object
          properties:
            filepaths:
              type: array
              items:
                type: string
                example: /uploads/archivo.nc
      400:
        description: Error en la subida
      409:
        description: Conflicto, el archivo ya existe
      500:
        description: Error interno
    """

    if 'files' not in request.files:
        return jsonify({"error": "No se encontraron archivos en la solicitud"}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "Lista de archivos vacía"}), 400

    saved_paths = []
    try:
        for file in files:
            if file.filename == '':
                continue  # Saltear archivos sin nombre

            if not allowed_file(file.filename):
                return jsonify({"error": f"Formato inválido: {file.filename}"}), 400

            unique_name = secure_filename(file.filename)
            upload_dir = current_app.config['UPLOAD_FOLDER']
            os.makedirs(upload_dir, exist_ok=True)

            filepath = os.path.join(upload_dir, unique_name)

            if os.path.exists(filepath):
                return jsonify({"error": f"El archivo {unique_name} ya existe."}), 409

            file.save(filepath)
            saved_paths.append(filepath)

        if not saved_paths:
            return jsonify({"error": "Ningún archivo válido fue subido"}), 400

        return jsonify({"filepaths": saved_paths}), 200

    except Exception as e:
        return jsonify({"error": f"Error al subir archivos: {str(e)}"}), 500
