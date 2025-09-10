# app.py
import os
import csv
from datetime import datetime
from flask import Flask, request, render_template, jsonify
import numpy as np
from tensorflow import keras



USE_GITHUB_AS_DB = os.getenv("USE_GITHUB_AS_DB", "false").lower() == "true"
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
DB_PATH = "predictions_log.csv"

app = Flask(__name__)


MODEL_PATH = os.getenv("MODEL_PATH", "model.keras")
model = keras.models.load_model(MODEL_PATH)

def predict_from_features(features):
    # features: list-like of length 4 (VWTI, SWTI, CWTI, EI)
    arr = np.array(features, dtype=float).reshape(1, -1)
    prob = float(model.predict(arr)[0][0])
    label = 1 if prob >= 0.5 else 0
    return {"probability": prob, "label": label}

# simple local logger (append to CSV)
def log_prediction(features, result, source="api"):
    header = ["timestamp", "source", "v_wti", "s_wti", "c_wti", "e_i", "probability", "label"]
    row = [datetime.utcnow().isoformat() + "Z", source] + [str(x) for x in features] + [str(result["probability"]), str(result["label"])]
    write_local_csv_row(DB_PATH, header, row)
    if USE_GITHUB_AS_DB and GITHUB_TOKEN and GITHUB_REPO:
        try:
            push_csv_to_github(DB_PATH, GITHUB_REPO, GITHUB_TOKEN)
        except Exception as e:
            app.logger.error("GitHub push failed: %s", e)

def write_local_csv_row(path, header, row):
    exists = os.path.exists(path)
    mode = "a"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)

# ---------------------------
# GitHub helpers (optional)
# ---------------------------
if USE_GITHUB_AS_DB:
    from github import Github
    def push_csv_to_github(local_path, repo_fullname, token, branch="main", commit_msg=None):
        """
        Push (create/update) the file at local_path to repo_fullname:path
        NOTE: repo_fullname is "owner/repo"
        """
        commit_msg = commit_msg or f"Update {local_path} ({datetime.utcnow().isoformat()}Z)"
        g = Github(token)
        repo = g.get_repo(repo_fullname)
        # read file
        with open(local_path, "rb") as fh:
            content = fh.read()
        try:
            contents = repo.get_contents(local_path, ref=branch)
            repo.update_file(contents.path, commit_msg, content, contents.sha, branch=branch)
        except Exception as e:
            # file may not exist -> create
            repo.create_file(local_path, commit_msg, content, branch=branch)

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        v = float(request.form["variance"])
        s = float(request.form["skewness"])
        c = float(request.form["kurtosis"])
        e = float(request.form["entropy"])
    except Exception as ex:
        return render_template("index.html", error="Invalid input: " + str(ex))
    features = [v, s, c, e]
    result = predict_from_features(features)
    log_prediction(features, result, source="web")
    return render_template("index.html", features=features, result=result)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST JSON: {"variance":..,"skewness":..,"kurtosis":..,"entropy":..}
    Response: {"probability":..,"label":0|1}
    """
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Send JSON body"}), 400
    try:
        features = [float(data[k]) for k in ("variance", "skewness", "kurtosis", "entropy")]
    except Exception as ex:
        return jsonify({"error":"Invalid/missing fields: " + str(ex)}), 400
    result = predict_from_features(features)
    log_prediction(features, result, source="api")
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    # For local dev only: flask run or python app.py
    app.run(host="0.0.0.0", port=port, debug=False)
