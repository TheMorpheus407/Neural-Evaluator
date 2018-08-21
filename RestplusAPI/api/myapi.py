from flask_restplus import Api

api = Api(version='0.1', title='Neural Evaluator', description='A Neural Network for evaluating the success of a pentest.')

@api.errorhandler
def std_handler(e):
    """standard error message"""
    return {'message': 'An unexpected error has occured. Please contact the support.'}, 500