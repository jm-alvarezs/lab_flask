#test individual functions
from flask import Flask, request
from kmeans import getKmeans
from bayes import getBayes
from knn import getKnn
from linear import getLinear
from linearbayes import getLinearBayes
from linearknn import getLinearKnn
import json
from json import JSONEncoder
import numpy

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

app = Flask(__name__)

@app.route('/test', methods=['GET'])
def home():
    return "Hello World"

@app.route('/kmeans', methods=['POST'])
def kmeans():
    prediction = getKmeans(request.json['index'], request.json['reaction'], request.json['correct'], request.json['error'])
    data = { 'prediction': prediction }
    result = json.dumps(data, cls=NumpyArrayEncoder)
    return result

@app.route('/bayes', methods=['POST'])
def bayes():
    prediction = getBayes(request.json['index'], request.json['reaction'], request.json['correct'], request.json['error'])
    return { 'prediction': prediction}

@app.route('/knn', methods=['POST'])
def knn():
    prediction = getKnn(request.json['index'], request.json['reaction'], request.json['correct'], request.json['error'])
    return { 'prediction': prediction}

@app.route('/linear', methods=['POST'])
def linear():
    prediction = getLinear(request.json['index'], request.json['reaction'], request.json['correct'], request.json['error'])
    return { 'prediction': prediction}

@app.route('/linear/bayes', methods=['POST'])
def linearBayes():
    prediction = getLinearBayes(request.json['index'], request.json['reaction'], request.json['correct'], request.json['error'])
    return { 'prediction': prediction}

@app.route('/linear/knn', methods=['POST'])
def linearKnn():
    prediction = getLinearKnn(request.json['index'], request.json['reaction'], request.json['correct'], request.json['error'])
    return { 'prediction': prediction}

app.run()