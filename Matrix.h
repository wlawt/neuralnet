#ifndef DEF_MATRIX
#define DEF_MATRIX

#include <vector>
#include <ostream>

class Matrix
{
public:
    Matrix();

    // Possible Parameteres
    Matrix(int height, int width);
    Matrix(std::vector<std::vector<double>> const &array);

    // Functions
    Matrix multiply(double const &value); // Scalar multiplication
    Matrix add(Matrix const &m) const;
    Matrix subtract(Matrix const &m) const;
    Matrix multiply(Matrix const &m) const;

    // Operators
    Matrix dot(Matrix const &m) const;
    Matrix transpose() const;

    // Activation function
    Matrix applyFunction(double (*function)(double)) const;

    // Print
    void print(std::ostream &flux) const;

private:
    std::vector<std::vector<double>> array;
    int height;
    int width;
};

std::ostream &operator<<(std::ostream &flux, Matrix const &m);

#endif