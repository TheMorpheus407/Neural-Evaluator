from flask_restplus import fields
from RestplusAPI.api.myapi import api

set = api.model('A new training set', {
    'filename': fields.String(required=True, description='The url to the website. Only needed as an identifier. No request will be issued.'),
    'raw_website': fields.String(required=True, description='A concatenation of headers and the body of the original non-attacked website. Please remove date headers! They irritate the network.'),
    'attacked_website': fields.String(required=True, description='A concatenation of headers and the body of the websites response when sending the payload. Please remove date headers! They irritate the network.'),
    'payload': fields.String(required=True, description='The payload sent to the website which led to the response in "attacked website"'),
    'successful': fields.Boolean(required=True, description='This indicates weather the payload was successful on given website.'),
})

set2 = api.model('A new training set', {
    'file': fields.String(required=True, description='The filename where the website is saved which has not received a payload.'),
    'attacked_file': fields.String(required=True, description='The filename where the website is saved which has received a payload.'),
    'payload': fields.Raw(required=True, description='The payload sent to the website which led to the response in "attacked website"'),
    'target': fields.String(required=True, description='This indicates where the website is vulnerable. Not yet implemented. Target = "-" indicates no vulnerability, another target indicates vulnerability.'),
    'method': fields.String(required=True, description='Currently unused. Will always contain "post"'),
})

list_of_sets = api.inherit('List of sets', {
    'items': fields.List(fields.Nested(set2))
})


query = api.model('A set to be evaluated', {
    'raw_website': fields.String(required=True, description='A concatenation of headers and the body of the original non-attacked website. Please remove date headers! They irritate the network.'),
    'attacked_website': fields.String(required=True, description='A concatenation of headers and the body of the websites response when sending the payload. Please remove date headers! They irritate the network.'),
    'payload': fields.String(required=True, description='The payload sent to the website which led to the response in "attacked website"')
})

query_answer = api.model('The answer to the query', {
    'successful': fields.Boolean(required=True, description='This indicates weather the payload was successful on given website.')
})

filename = api.model('The name of the file', {
    'name': fields.String(required=True, description='The filename of the ressource to be changed.'),
})

category = api.model('Product category', {
    'id': fields.Integer(readOnly=True, description='The identifier of the category'),
    'name': fields.String(required=True, description='Category name'),
})

pagination = api.model('One page of products', {
    'page': fields.Integer(description='Current page'),
    'pages': fields.Integer(description='Total pages'),
    'items_per_page': fields.Integer(description='Items per page'),
    'total_items': fields.Integer(description='Total amount of items')
})
