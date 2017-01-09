package utils

import (
	"log"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
)

// Gemm computes
//  C = alpha * A * B + beta * C,
// where A, B, and C are dense matrices, and alpha and beta are scalars.
// tA and tB specify whether A or B are transposed.
func gocaffeGemm(tA, tB blas.Transpose, alpha float64, a, b blas64.General, beta float64, c blas64.General) {
	blas64.Gemm(tA, tB, alpha, a, b, beta, c)
}

// Gemv computes
//  y = alpha * A * x + beta * y,   if t == blas.NoTrans,
//  y = alpha * A^T * x + beta * y, if t == blas.Trans or blas.ConjTrans,
// where A is an m×n dense matrix, x and y are vectors, and alpha and beta are scalars.
func gocaffeGemv(t blas.Transpose, alpha float64, a blas64.General, x blas64.Vector, beta float64, y blas64.Vector) {
	blas64.Gemv(t, alpha, a, x, beta, y)
}

// Axpy adds x scaled by alpha to y:
//  y[i] += alpha*x[i] for all i.
func gocaffeAxpy(n int, alpha float64, x, y blas64.Vector) {
	blas64.Axpy(n, alpha, x, y)
}

// Scal scales the vector x by alpha:
//  x[i] *= alpha for all i.
//
// Scal will panic if the vector increment is negative.
func gocaffeScal(n int, alpha float64, x blas64.Vector) {
	blas64.Scal(n, alpha, x)
}
