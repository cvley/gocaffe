package utils

import (
	"testing"
)

func TestIm2Col(t *testing.T) {
	input := []float64{1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0}

	output, err := Im2col(input, 1, 3, 3, 1, 1, 0, 0, 1, 1, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%+v", output)

	in, err := Col2im(output, 1, 3, 3, 1, 1, 0, 0, 1, 1, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(input) != len(in) {
		t.Fatal("col2im fail: mismatch length")
	}
	for i, v := range in {
		if v != input[i] {
			t.Fatal("col2im fail: mismatch value")
		}
	}
}
