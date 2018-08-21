from flask import request
from RestplusAPI.api.myapi import api
from flask_restplus import Resource
from flask_restplus import Namespace
from RestplusAPI.api.api_endpoints.api_definition import query, query_answer
import RestplusAPI.api.query as apiquery

namespace = Namespace('query/', description='Query the network.')

@api.doc(description="Query the neural network.")
@namespace.route('/')
class Query(Resource):
    @api.expect(query)
    @api.marshal_with(query_answer)
    def post(self):
        return {"successful": apiquery.query(request.json)}, 200








