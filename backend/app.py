from flask import Flask
from flasgger import Swagger
from config import Config
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)
app.config.from_object(Config)

from routes import upload, process
app.register_blueprint(upload.bp)
app.register_blueprint(process.bp)

if __name__ == "__main__":
    app.run(debug=True)
