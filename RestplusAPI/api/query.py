import RestplusAPI.NeuralEvaluator.neuralevaluator_cnn as Evaluator

def query(json_params):
    """queries the network. Make sure that the json follows the Swagger definition"""
    return Evaluator.predict(json_params["raw_website"], json_params["attacked_website"], payload=json_params["payload"])
