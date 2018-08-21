from flask import request
from RestplusAPI.api.myapi import api
from flask_restplus import Resource
from flask_restplus import Namespace
from RestplusAPI.api.api_endpoints.api_definition import set, list_of_sets, filename
import RestplusAPI.api.crud as apicrud

namespace = Namespace('sets/', description='Enhance the training dataset and retrain the network.')

@api.doc(description="Insert one new dataset. Will NOT initiate retraining.")
@namespace.route('/addset/')
class Offer(Resource):
    @api.expect(set)
    def post(self):
        apicrud.create_set(request.json["url"], request.json["raw_website"], request.json["attacked_website"], request.json["payload"], request.json["successful"])
        return None, 200

@api.doc(description="Read all training meta data")
@namespace.route('/getsets/')
class SetItem(Resource):
    @api.marshal_with(list_of_sets)
    def get(self):
        return apicrud.read_sets()

@api.doc(description="Delete a set by filename - see attacked_file in /getsets/. Make sure to only include the filename, exclude any of the folders. Bsp: xss.php_1.raw NOT: Sites/xss.php_1.raw")
@namespace.route('/deleteset/<filename>')
class DeleteSet(Resource):
    def get(self, filename):
        return apicrud.delete_set(filename)








