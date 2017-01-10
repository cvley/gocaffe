package blob

import (
	"testing"
)

func TestNewBlob(t *testing.T) {
	shape := []int32{1, 1, 1, 1}
	b := New([]int32{1, 1, 1, 1})
	if len(b.shape) != len(shape) {
		t.Fatal("shape mismatch")
	}
}
