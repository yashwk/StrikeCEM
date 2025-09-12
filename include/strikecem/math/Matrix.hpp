#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include "Vector.hpp"

namespace la {

    template<typename T>
    class Matrix {
        size_t rows, cols;
        std::vector<T> data; // Stored in row-major order

    public:
        // --- Constructors ---
        Matrix(size_t r, size_t c) : rows(r), cols(c), data(r*c, T(0)) {}

        // --- Accessors ---
        T& operator()(size_t r, size_t c) { return data[r*cols + c]; }
        const T& operator()(size_t r, size_t c) const { return data[r*cols + c]; }
        size_t rowCount() const { return rows; }
        size_t colCount() const { return cols; }

        // --- Static Methods ---
        static Matrix identity(size_t n) {
            Matrix I(n, n);
            for (size_t i = 0; i < n; ++i) I(i, i) = T(1);
            return I;
        }

        // --- Core Matrix Operations ---
        T determinant() const {
            #ifndef NDEBUG
            if (rows != cols) throw std::runtime_error("Determinant requires a square matrix");
            #endif
            if (rows == 2) {
                return (*this)(0,0) * (*this)(1,1) - (*this)(0,1) * (*this)(1,0);
            }
            if (rows == 3) {
                return (*this)(0,0) * ((*this)(1,1) * (*this)(2,2) - (*this)(1,2) * (*this)(2,1)) -
                       (*this)(0,1) * ((*this)(1,0) * (*this)(2,2) - (*this)(1,2) * (*this)(2,0)) +
                       (*this)(0,2) * ((*this)(1,0) * (*this)(2,1) - (*this)(1,1) * (*this)(2,0));
            }
            throw std::runtime_error("Determinant not implemented for N>3 matrices");
        }

        Matrix inverse() const {
            #ifndef NDEBUG
            if (rows != cols) throw std::runtime_error("Inverse requires a square matrix");
            #endif
            T det = determinant();
            if (std::abs(det) < T(1e-9)) throw std::runtime_error("Matrix is singular and cannot be inverted");

            T invDet = T(1) / det;
            Matrix inv(rows, cols);

            if (rows == 2) {
                inv(0,0) = (*this)(1,1) * invDet;
                inv(0,1) = -(*this)(0,1) * invDet;
                inv(1,0) = -(*this)(1,0) * invDet;
                inv(1,1) = (*this)(0,0) * invDet;
                return inv;
            }
            if (rows == 3) {
                // Tedious but direct implementation of 3x3 inverse
                inv(0,0) = ((*this)(1,1)*(*this)(2,2) - (*this)(2,1)*(*this)(1,2)) * invDet;
                inv(0,1) = ((*this)(0,2)*(*this)(2,1) - (*this)(0,1)*(*this)(2,2)) * invDet;
                inv(0,2) = ((*this)(0,1)*(*this)(1,2) - (*this)(0,2)*(*this)(1,1)) * invDet;
                inv(1,0) = ((*this)(1,2)*(*this)(2,0) - (*this)(1,0)*(*this)(2,2)) * invDet;
                inv(1,1) = ((*this)(0,0)*(*this)(2,2) - (*this)(0,2)*(*this)(2,0)) * invDet;
                inv(1,2) = ((*this)(1,0)*(*this)(0,2) - (*this)(0,0)*(*this)(1,2)) * invDet;
                inv(2,0) = ((*this)(1,0)*(*this)(2,1) - (*this)(2,0)*(*this)(1,1)) * invDet;
                inv(2,1) = ((*this)(2,0)*(*this)(0,1) - (*this)(0,0)*(*this)(2,1)) * invDet;
                inv(2,2) = ((*this)(0,0)*(*this)(1,1) - (*this)(1,0)*(*this)(0,1)) * invDet;
                return inv;
            }
            throw std::runtime_error("Inverse not implemented for N>3 matrices");
        }

        // --- Matrix-Matrix Multiplication ---
        Matrix operator*(const Matrix& other) const {
            #ifndef NDEBUG
            if (cols != other.rows) throw std::runtime_error("Matrix dimension mismatch for multiplication");
            #endif
            Matrix result(rows, other.cols);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < other.cols; ++j) {
                    T sum = T(0);
                    for (size_t k = 0; k < cols; ++k) {
                        sum += (*this)(i, k) * other(k, j);
                    }
                    result(i, j) = sum;
                }
            }
            return result;
        }
    };

    // --- Free Functions ---
    template<typename T>
    inline Vector<T> matvec(const Matrix<T>& A, const Vector<T>& x) {
        #ifndef NDEBUG
        if (A.colCount() != x.size()) throw std::runtime_error("Matrix-vector size mismatch");
        #endif
        Vector<T> res(A.rowCount());
        for (size_t i = 0; i < A.rowCount(); ++i) {
            T sum = T(0);
            for (size_t j = 0; j < A.colCount(); ++j)
                sum += A(i, j) * x[j];
            res[i] = sum;
        }
        return res;
    }

    template<typename T>
    inline Matrix<T> transpose(const Matrix<T>& A) {
        Matrix<T> T_mat(A.colCount(), A.rowCount());
        for (size_t i = 0; i < A.rowCount(); ++i)
            for (size_t j = 0; j < A.colCount(); ++j)
                T_mat(j, i) = A(i, j);
        return T_mat;
    }

    // --- Type Aliases ---
    using MatrixF = Matrix<float>;
    using MatrixD = Matrix<double>;

} // namespace la

