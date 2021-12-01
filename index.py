from flask import request
from kmeans import getKmeans
from bayes import getBayes
from knn import getKnn
from linear import getLinear
from linearbayes import getLinearBayes
from linearknn import getLinearKnn

@app.route('/kmeans', methods=['GET','POST'])
def home():
    pca = PCA()
    prediction = getKmeans(request.indice, request.reaction, request.correct, request.error)
    return prediction


@app.route('/bayes', methods=['GET','POST'])
def home():
    pca = PCA()
    prediction = getBayes(request.indice, request.reaction, request.correct, request.error, request.nechapi)
    return prediction

@app.route('/knn', methods=['GET','POST'])
def home():
    pca = PCA()
    prediction = getKnn(request.indice, request.reaction, request.correct, request.error, request.nechapi)
    return prediction

@app.route('/linear', methods=['GET','POST'])
def home():
    pca = PCA()
    prediction = getLinear(request.anger, request.sensation, request.emotional, request.sociability, request.motivation, request.col)
    return prediction

@app.route('/linear/bayes', methods=['GET','POST'])
def home():
    pca = PCA()
    prediction = getLinearBayes(request.anger, request.sensation, request.emotional, request.sociability, request.motivation, request.col)
    return prediction

@app.route('/linear/knn', methods=['GET','POST'])
def home():
    pca = PCA()
    prediction = getLinearKnn(request.anger, request.sensation, request.emotional, request.sociability, request.motivation, request.col)
    return prediction

app.run()