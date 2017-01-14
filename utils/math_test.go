package utils

import (
	"testing"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
)

func TestGemm(t *testing.T) {
	a := blas64.General{
		Rows:   3,
		Cols:   2,
		Stride: 2,
		Data:   []float64{1.0, 0.0, 1.0, 0.0, 2.0, 0.0},
	}

	b := blas64.General{
		Rows:   2,
		Cols:   2,
		Stride: 2,
		Data:   []float64{2.0, 0.0, 0.0, 2.0},
	}

	c := blas64.General{
		Rows:   3,
		Cols:   2,
		Stride: 2,
		Data:   make([]float64, 6),
	}

	gocaffeGemm(blas.NoTrans, blas.NoTrans, float64(1.0), a, b, float64(0), c)
	t.Logf("%+v", c)
}
