import os
import logging
import json
import numpy
import pandas as pd
import joblib
from azureml.core.model import Model


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "/var/azureml-app/azureml-models/classifier/1/lg_pipeline.pkl"
    )

    #model_path = Model.get_model_path('lg_pipeline.pkl')
    #model_path = os.path.join(model_dir, 'lg_pipeline.pkl')
    print(f"Loading model from {model_path}")
    
    #os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model/classifier.pkl")

    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)
    logging.debug("Init complete")


data_types = {'uuid': 'object',
 'default': 'float64',
 'account_amount_added_12_24m': 'int64',
 'account_days_in_dc_12_24m': 'float64',
 'account_days_in_rem_12_24m': 'float64',
 'account_days_in_term_12_24m': 'float64',
 'account_incoming_debt_vs_paid_0_24m': 'float64',
 'account_status': 'float64',
 'account_worst_status_0_3m': 'float64',
 'account_worst_status_12_24m': 'float64',
 'account_worst_status_3_6m': 'float64',
 'account_worst_status_6_12m': 'float64',
 'age': 'int64',
 'avg_payment_span_0_12m': 'float64',
 'avg_payment_span_0_3m': 'float64',
 'merchant_category': 'object',
 'merchant_group': 'object',
 'has_paid': 'bool',
 'max_paid_inv_0_12m': 'float64',
 'max_paid_inv_0_24m': 'float64',
 'name_in_email': 'object',
 'num_active_div_by_paid_inv_0_12m': 'float64',
 'num_active_inv': 'int64',
 'num_arch_dc_0_12m': 'int64',
 'num_arch_dc_12_24m': 'int64',
 'num_arch_ok_0_12m': 'int64',
 'num_arch_ok_12_24m': 'int64',
 'num_arch_rem_0_12m': 'int64',
 'num_arch_written_off_0_12m': 'float64',
 'num_arch_written_off_12_24m': 'float64',
 'num_unpaid_bills': 'int64',
 'status_last_archived_0_24m': 'int64',
 'status_2nd_last_archived_0_24m': 'int64',
 'status_3rd_last_archived_0_24m': 'int64',
 'status_max_archived_0_6_months': 'int64',
 'status_max_archived_0_12_months': 'int64',
 'status_max_archived_0_24_months': 'int64',
 'recovery_debt': 'int64',
 'sum_capital_paid_account_0_12m': 'int64',
 'sum_capital_paid_account_12_24m': 'int64',
 'sum_paid_inv_0_12m': 'int64',
 'time_hours': 'float64',
 'worst_status_active_inv': 'float64'}


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("model 1: request received")
    data1 = json.loads(json.loads(raw_data)["data"][0])
    #data = numpy.array(data)

    columns = [key for key,val in data1.items()]
    vals = [[val] for key,val in data1.items()]
    
    for i in range(len(vals)):
        if vals[i] == [None]:
            vals[i] = [numpy.nan]

    data = pd.DataFrame(numpy.array(vals).T,columns=columns)
    data = data.astype(data_types)

    result = model.predict_proba(data)
    logging.info("Request processed")
    return result.tolist()