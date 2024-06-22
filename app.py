from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS, cross_origin
import os
import time
import urllib.request
from flask import send_from_directory

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/')
def dafault_route():
    return 'API'


@app.route('/upload_data', methods=['POST'])
@cross_origin()
def uploadimg():
    for fname in request.files:
        f = request.files.get(fname)
        print(f)
        # milliseconds = int(time.time() * 1000)
        # filename = str(milliseconds)
        in_filename = f"./uploads/report.html"
        f.save(in_filename)
    return "load data ok"


@app.route('/uploads/<path:path>')
def send_photo(path):
    return send_from_directory('uploads', path)


@app.route("/textVis", methods=['POST'])
def textVis():
    pass
    import textVisLib
    textVisLib.save_data()
    out_data={}
    out_data['a1'] = "http://localhost:5000/uploads/TFD_risk.png"
    out_data['a2'] = "http://localhost:5000/uploads/tSNE_SVD_risk.png"
    out_data['a3'] = "http://localhost:5000/uploads/umap_CV_risk.png"
    out_data['a4'] = "http://localhost:5000/uploads/tSNE_SVD_bdu.png"
    out_data['a5'] = "http://localhost:5000/uploads/umap_CV_bdu.png"
    return out_data

@app.route("/clear_db", methods=['GET'])
def clear_db():
    return "ok clear_db"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000")