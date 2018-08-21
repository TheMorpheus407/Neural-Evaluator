import sys
import os
this_dir = os.getcwd()
parent_dir = os.path.abspath('..')
if not this_dir in sys.path:
    sys.path.append(this_dir)
if not parent_dir in sys.path:
    sys.path.append(parent_dir)


from flask import Flask, Blueprint
import RestplusAPI.settings as settings
from RestplusAPI.api.myapi import api
from RestplusAPI.api.api_endpoints.endpoints.sets import namespace as setnamespace
from RestplusAPI.api.api_endpoints.endpoints.retrain import namespace as retrain_namespace
from RestplusAPI.api.api_endpoints.endpoints.query import namespace as query_namespace

app = Flask(__name__)

def configure_app(app):
    """Set required app variables"""
    app.config['SWAGGER_UI_DOC_EXPANSION'] = settings.RESTPLUS_SWAGGER_EXPANSION
    app.config['RESTPLUS_VALIDATE'] = settings.RESTPLUS_VAL
    app.config['RESTPLUS_MASK_SWAGGER'] = settings.RESTPLUS_MASK_SWAGGER


def init_app(app):
    """Register config and blueprints"""
    configure_app(app)
    blueprint = Blueprint('api', __name__, url_prefix='/api')
    api.init_app(blueprint)
    api.add_namespace(setnamespace)
    api.add_namespace(retrain_namespace)
    api.add_namespace(query_namespace)
    app.register_blueprint(blueprint)


def main():
    """configure, then start the flask app"""
    init_app(app)
    app.run(debug=settings.FLASK_DEBUG, threaded=settings.FLASK_THREADED, port=1337, host="0.0.0.0")

if __name__ == "__main__":
    main()
