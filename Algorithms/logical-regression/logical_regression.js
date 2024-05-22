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


module.exports = LogisticRegression;
