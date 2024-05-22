// test_logistic_regression.js

/*
 * Test Cases for Logistic Regression Implementation in JavaScript
 * 
 * This file includes:
 * 1. Initialization of training data.
 * 2. Creation of Logistic Regression model instance.
 * 3. Training the model with the data.
 * 4. Making predictions with the model.
 * 5. Logging the results.
 */

const LogisticRegression = require('./logistic_regression');

// Test cases
function runTestCases() {
    const X = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ];
    const y = [0, 0, 0, 1];  // AND logic gate

    const model = new LogisticRegression(learningRate = 0.1, iterations = 1000);
    model.fit(X, y);

    const predictions = model.predict(X);

    console.log('Test cases:');
    console.log('Input: [0, 0], Expected Output: 0, Predicted Output:', predictions[0]);
    console.log('Input: [0, 1], Expected Output: 0, Predicted Output:', predictions[1]);
    console.log('Input: [1, 0], Expected Output: 0, Predicted Output:', predictions[2]);
    console.log('Input: [1, 1], Expected Output: 1, Predicted Output:', predictions[3]);
}

// Run test cases
runTestCases();
