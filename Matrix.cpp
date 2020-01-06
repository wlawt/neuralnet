#include "Matrix.h"
#include <assert.h>
#include <sstream>

// init
Matrix::Matrix() {}
Matrix::Matrix(int height, int width)
{
    this->height = height;
    this->width = width;
    this->array = std::vector<std::vector<double>>(height, std::vector<double>(width));
}

Matrix::Matrix(std::vector<std::vector<double>> const &array)
{
    assert(array.size() != 0); // Check to see if the matrix always has something
    this->height = array.size();
    this->width = array[0].size();
    this->array = array;
}

// Scalar multiplication
Matrix Matrix::multiply(double const &value)
{
    int i, j;

    Matrix result(height, width);

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            result.array[i][j] = array[i][j] * value;
        }
    }

    return result;
}

// Matrix addition
Matrix Matrix::add(Matrix const &m) const
{
    assert(height == m.height && width == m.width);
    int i, j;

    Matrix result(height, width);

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            result.array[i][j] = array[i][j] + m.array[i][j];
        }
    }

    return result;
}

// Matrix subtraction
Matrix Matrix::subtract(Matrix const &m) const
{
    assert(height == m.height && width == m.width);
    int i, j;

    Matrix result(height, width);

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            result.array[i][j] = array[i][j] - m.array[i][j];
        }
    }

    return result;
}

// Matrix multiplication
Matrix Matrix::multiply(Matrix const &m) const
{
    assert(height == m.height && width == m.width);
    int i, j;

    Matrix result(height, width);

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            result.array[i][j] = array[i][j] * m.array[i][j];
        }
    }

    return result;
}

// Dot product
Matrix Matrix::dot(Matrix const &m) const
{
    assert(width == m.height);

    int i, j, h, mwidth = m.width;
    double w = 0;

    Matrix result(height, mwidth);

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < mwidth; j++)
        {
            for (h = 0; h < width; h++)
            {
                w += array[i][h] * m.array[h][j];
            }
            result.array[i][j] = w;
            w = 0;
        }
    }

    return result;
}

// Matrix tranpose
Matrix Matrix::transpose() const
{
    int i, j;

    Matrix result(height, width);

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            result.array[i][j] = array[j][i];
        }
    }

    return result;
}

// Activation function
Matrix Matrix::applyFunction(double (*function)(double)) const
{
    Matrix result(height, width);

    int i, j;

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            result.array[i][j] = (*function)(array[i][j]);
        }
    }

    return result;
}

// Print
void Matrix::print(std::ostream &flux) const
{
    int i, j;
    int maxLength[width] = {};
    std::stringstream ss;

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            // Concat vector elements into string
            ss << array[i][j];

            // j of the vector is the width
            if (maxLength[j] < ss.str().size())
            {
                maxLength[j] = ss.str().size();
            }

            // To string
            ss.str(std::string());
        }
    }

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            flux << array[i][j];
            ss << array[i][j];

            for (int k = 0; k < maxLength[j] - ss.str().size() + 1; k++)
            {
                flux << " ";
            }

            ss.str(std::string());
        }

        flux << std::endl; // New line
    }
}

// Run
std::ostream &operator<<(std::ostream &flux, Matrix const &m)
{
    m.print(flux);
    return flux;
}