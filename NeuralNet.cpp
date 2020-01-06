#include <vector>
#include <fstream>
#include <cmath>
#include <time.h>
#include <iostream>
#include <stdlib.h>

#include "Matrix.h"

using namespace std;

void loadingTraining(const char *filename, vector<vector<double>> &input, vector<vector<double>> &output)
{
    int trainingSize = 946;
    input.resize(trainingSize);
    output.resize(trainingSize);

    // file exist
    ifstream file(filename);

    if (file)
    {
        string line;
        int n;

        // Load training images
        for (int i = 0; i < trainingSize; i++)
        {
            // height pixels
            for (int h = 0; h < 32; h++)
            {
                getline(file, line);

                // width pixels
                for (int w = 0; w < 32; w++)
                {
                    input[i].push_back(atoi(line.substr(w, 1).c_str()));
                }
            }

            getline(file, line);

            // Output neurons 0 - 9
            output[i].resize(10);

            // Get res
            n = atoi(line.substr(0, 1).c_str());
            output[i][n] = 1;
        }
    }
}

Matrix X, W1, H, W2, Y, B1, B2, Y2, dJdB1, dJdB2, dJdW1, dJdW2;
double learningRate;

// RNG
double random(double x)
{
    return (double)(rand() % 10000 + 1) / 10000 - 0.5;
}

/* 
    Architecture
 */
void init(int inputNeuron, int hiddenNeuron, int OutputNeuron, double rate)
{
    learningRate = rate;

    W1 = Matrix(inputNeuron, hiddenNeuron);
    W2 = Matrix(hiddenNeuron, OutputNeuron);
    B1 = Matrix(1, inputNeuron);
    B2 = Matrix(1, OutputNeuron);

    W1 = W1.applyFunction(random);
    W2 = W2.applyFunction(random);
    B1 = B1.applyFunction(random);
    B2 = B2.applyFunction(random);
}

// Sigmoid activation function
double sigmoid(double x)
{
    return (1 / 1 + exp(-x));
}

/* 
    Computations
 */
Matrix computeOutput(vector<double> input)
{
    // Row matrix
    X = Matrix({input});

    H = X.dot(W1).add(B1).applyFunction(sigmoid);
    Y = H.dot(W2).add(B2).applyFunction(sigmoid);

    return Y;
}

// Derivative of sigmoid activation function
double sigmoidPrime(double x)
{
    return (exp(-x) / 1 + pow(exp(-x), 2));
}

/* 
    Backprop - Gradient descent 
 */
Matrix learn(vector<double> expectedOutput)
{
    // Row matrix
    Y2 = Matrix({expectedOutput});

    // Gradients
    dJdB2 = Y.subtract(Y2).multiply(H.dot(W2).add(B2).applyFunction(sigmoidPrime));
    dJdB1 = dJdB2.dot(W2.transpose()).multiply(X.dot(W1).add(B1).applyFunction(sigmoidPrime));
    dJdW2 = H.transpose().dot(dJdB2);
    dJdW1 = X.transpose().dot(dJdB1);

    // Backprop
    W1 = W1.subtract(dJdW1.multiply(learningRate));
    W2 = W2.subtract(dJdW2.multiply(learningRate));
    B1 = B1.subtract(dJdB1.multiply(learningRate));
    B2 = B2.subtract(dJdB2.multiply(learningRate));

    return Y2;
}

/* 
    If sigmoid never reaches 0.0 or 1.0, we'd consider rounding based 
    on these conditions
 */
double stepFunction(double x)
{
    if (x > 0.9)
    {
        return 1.0;
    }

    if (x < 0.1)
    {
        return 0.0;
    }

    return x;
}

/* 
    Run neural net
 */
int main(int argc, char *argv[])
{
    // Generate random weights
    srand(time(NULL));

    std::vector<std::vector<double>> inputVector, outputVector;

    // Load training file called "training"
    loadingTraining("training", inputVector, outputVector);

    /* 
        1024 - 32 * 32 image size
        15 hidden neurons
        10 output neurons
        0.7 learning rate
     */
    init(1024, 15, 10, 0.7);

    // Train for 30 epochs
    for (int i = 0; i < 30; i++)
    {
        // Staying within size of the input vector
        for (int j = 0; j < inputVector.size(); j++)
        {
            // Forward prop
            computeOutput(inputVector[j]);

            // Backprop
            learn(outputVector[j]);
        }

        // Log
        cout << "#" << i + 1 << "/30" << endl;
    }

    // Test
    cout << endl
         << "expected output : actual output" << endl;

    for (int i = inputVector.size(); i < inputVector.size(); i++)
    {
        // Test on last 10 samples
        for (int j = 0; j < 10; j++)
        {
            cout << outputVector[i][j] << " ";
        }

        cout << ": " << computeOutput(inputVector[i]).applyFunction(stepFunction) << endl;
    }
}
