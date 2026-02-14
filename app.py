from flask import Flask, render_template, request
import numpy as np
import pickle
import math

app = Flask(__name__)

# =========================================================
# DEFINE KNN CLASS (Required for Pickle)
# =========================================================
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = []
        for x in X:
            distances = []
            for i in range(len(self.X_train)):
                dist = self.distance(x, self.X_train[i])
                distances.append((dist, self.y_train[i]))

            distances.sort(key=lambda x: x[0])
            k_labels = [label for _, label in distances[:self.k]]
            prediction = max(set(k_labels), key=k_labels.count)
            predictions.append(prediction)

        return np.array(predictions)


# =========================================================
# DEFINE GAUSSIAN NB CLASS (Required for Pickle)
# =========================================================
class GaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.prior = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.prior[c] = len(X_c) / len(X)

    def pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var + 1e-9))
        denominator = np.sqrt(2 * np.pi * var + 1e-9)
        return numerator / denominator

    def predict(self, X):
        predictions = []

        for x in X:
            posteriors = []

            for c in self.classes:
                prior = math.log(self.prior[c])
                conditional = np.sum(np.log(self.pdf(c, x) + 1e-9))
                posteriors.append(prior + conditional)

            predictions.append(self.classes[np.argmax(posteriors)])

        return np.array(predictions)


# =========================================================
# LOAD PICKLE MODELS (Now Works)
# =========================================================
with open("KNN.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("NB.pkl", "rb") as f:
    nb_model = pickle.load(f)

# Class label mapping
class_names = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

# =========================================================
# ROUTES
# =========================================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    sl = float(request.form["sl"])
    sw = float(request.form["sw"])
    pl = float(request.form["pl"])
    pw = float(request.form["pw"])
    algo = request.form["algo"]

    input_data = np.array([[sl, sw, pl, pw]])

    if algo == "knn":
        prediction = knn_model.predict(input_data)[0]
    else:
        prediction = nb_model.predict(input_data)[0]

    result = class_names[int(prediction)]

    return render_template("index.html", prediction=result, algorithm=algo.upper())


if __name__ == "__main__":
    app.run(debug=True)
