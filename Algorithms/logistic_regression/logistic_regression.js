// logistic_regression.js

/*
 * Logistic Regression Implementation in JavaScript
 * 
 * This implementation includes:
 * 1. The Logistic Regression class.
 * 2. Functions for model training.
 * 3. Functions for model prediction.
 */

/**
 * Sigmoid function
 * @param {number} z - The input value.
 * @returns {number} - The output of the sigmoid function.
 */
function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
}

/**
 * Logistic Regression Class
 */
class LogisticRegression {
    constructor(learningRate = 0.01, iterations = 1000) {
        this.learningRate = learningRate;
        this.iterations = iterations;
        this.weights = null;
        this.bias = 0;
    }

    /**
     * Fits the logistic regression model to the training data
     * @param {Array} X - The training features.
     * @param {Array} y - The training labels.
     */
    fit(X, y) {
        const m = X.length;
        const n = X[0].length;

        // Initialize weights
        this.weights = new Array(n).fill(0);

        for (let iter = 0; iter < this.iterations; iter++) {
            let linearModel = new Array(m).fill(0);
            let predictions = new Array(m).fill(0);
            let dw = new Array(n).fill(0);
            let db = 0;

            // Compute linear model and predictions
            for (let i = 0; i < m; i++) {
                linearModel[i] = this.bias;
                for (let j = 0; j < n; j++) {
                    linearModel[i] += this.weights[j] * X[i][j];
                }
                predictions[i] = sigmoid(linearModel[i]);
            }

            // Compute gradients
            for (let i = 0; i < m; i++) {
                const error = predictions[i] - y[i];
                for (let j = 0; j < n; j++) {
                    dw[j] += X[i][j] * error;
                }
                db += error;
            }

            // Update weights and bias
            for (let j = 0; j < n; j++) {
                this.weights[j] -= (this.learningRate * dw[j]) / m;
            }
            this.bias -= (this.learningRate * db) / m;
        }
    }

    /**
     * Predicts the labels for the given input data
     * @param {Array} X - The input features.
     * @returns {Array} - The predicted labels.
     */
    predict(X) {
        const m = X.length;
        let predictions = new Array(m).fill(0);

        for (let i = 0; i < m; i++) {
            let linearModel = this.bias;
            for (let j = 0; j < X[0].length; j++) {
                linearModel += this.weights[j] * X[i][j];
            }
            predictions[i] = sigmoid(linearModel) >= 0.5 ? 1 : 0;
        }

        return predictions;
    }
}

module.exports = LogisticRegression;