from flask import request
from RestplusAPI.api.myapi import api
from flask_restplus import Resource
from flask_restplus import Namespace
from RestplusAPI.api.api_endpoints.api_definition import set, list_of_sets, filename
import RestplusAPI.api.retrain as apiretrain

namespace = Namespace('retrain/', description='Retrain the network.')

@api.doc(description="Retrain the neural network. Will use up plenty of time! Do NOT use twice. Will return 'ok' when finished.")
@namespace.route('/')
class Retrain(Resource):
    def post(self):
        apiretrain.retrain()
        return "ok", 200








