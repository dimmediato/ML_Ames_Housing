"""
AWS Lambda handler: serve Ames house-price predictions.
Downloads the trained model from S3 on cold start, predicts per request.
"""
import os
import io
import json
import boto3
import joblib
import numpy as np
import pandas as pd
import base64

# --- configuration via environment variables (set on the Lambda function) ---
MODEL_BUCKET = os.environ["MODEL_BUCKET"]
MODEL_KEY = os.environ.get("MODEL_KEY", "ames_model.joblib")

# Columns the notebook cast to non-numeric before training. Serving MUST match,
# or the encoders silently mismatch categories and predictions degrade.
RAW_OBJECT_COLS = ["YrSold", "MoSold", "YearRemodAdd", "YearBuilt", "MSSubClass"]

# --- COLD-START CODE: runs once when the container boots, NOT per request ---
# Loading the model here (outside the handler) means it's fetched and
# deserialized a single time, then reused across all warm invocations.
s3 = boto3.client("s3")
_obj = s3.get_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
model = joblib.load(io.BytesIO(_obj["Body"].read()))   # load straight to memory; Lambda's disk is read-only except /tmp
EXPECTED = list(model.feature_names_in_)               # the 79 columns the pipeline expects


def prep_raw(features: dict) -> pd.DataFrame:
    """Rebuild one raw house record into the exact shape the pipeline was trained on."""
    row = pd.DataFrame([features])
    row = row.reindex(columns=EXPECTED)      # add any missing cols as NaN, drop unknown extras, fix order
    row = row.replace({None: np.nan})        # JSON null -> NaN so the imputers handle it
    for c in RAW_OBJECT_COLS:
        row[c] = row[c].astype("object")
    row["GarageYrBlt"] = row["GarageYrBlt"].astype("str")
    return row

def _response(status, payload):
    """Lambda-proxy response shape that API Gateway (payload format 2.0) understands."""
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def _extract_features(event):
    """Pull the features dict out of either call style.

    - API Gateway HTTP API: the JSON body arrives as a STRING under event['body'].
    - Direct boto3 invoke:  the event IS the payload; features sit at event['features'].
    """
    if "body" in event:                       # came through API Gateway
        raw = event["body"]
        if event.get("isBase64Encoded"):
            raw = base64.b64decode(raw).decode("utf-8")
        payload = json.loads(raw) if isinstance(raw, str) else (raw or {})
    else:                                      # direct invoke
        payload = event
    return payload["features"]


def handler(event, context):
    """Returns the predicted price, wrapped so API Gateway can serve it over HTTP."""
    try:
        features = _extract_features(event)
    except (KeyError, TypeError, ValueError):   # missing key, non-dict, or malformed JSON
        return _response(400, {"error": "Request must include a 'features' object."})

    price = float(model.predict(prep_raw(features))[0])
    return _response(200, {"predicted_price": round(price, 2)})