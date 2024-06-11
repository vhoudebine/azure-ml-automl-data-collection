
import logging
import os
import json
import mlflow
from io import StringIO
from mlflow.pyfunc.scoring_server import infer_and_parse_json_input, predictions_to_json
from azureml.ai.monitoring import Collector
import pandas as pd



def init():
    global model
    global input_schema
    global inputs_collector, outputs_collector
    
    inputs_collector = Collector(name='model_inputs')          
    outputs_collector = Collector(name='model_outputs')
    # "model" is the path of the mlflow artifacts when the model was registered. For automl
    # models, this is generally "mlflow-model".
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "mlflow-model")
    model = mlflow.pyfunc.load_model(model_path)
    input_schema = model.metadata.get_input_schema()


def run(raw_data):
    json_data = json.loads(raw_data)
    if "input_data" not in json_data.keys():
        raise Exception("Request must contain a top level key named 'input_data'")

    serving_input = json.dumps(json_data["input_data"])
    data = infer_and_parse_json_input({'dataframe_split':json_data["input_data"]}, input_schema)
    
    #Logging inputs, running inference and logging outputs
    context = inputs_collector.collect(data)
    predictions = model.predict(data)
    
    output_df = pd.DataFrame(predictions, columns=['prediction'])
    outputs_collector.collect(output_df, context)
    
    return output_df.to_dict()
