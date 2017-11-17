package array

import (
	"testing"
)

func TestShape(t *testing.T) {
	invalidShapes := []Shape{
		Shape{0, 1},
		Shape{-1, 1},
	}

	for _, s := range invalidShapes {
		if s.isValid() {
			t.Fatal("shape should be invalid")
		}
	}

	validShape := []Shape{
		Shape{1, 3},
		Shape{2, 4},
		Shape{3},
	}

	for _, s := range validShape {
		if !s.isValid() {
			t.Fatal("shape should be valid")
		}
		t.Logf("Shape:%s Cap: %d\n", s, s.Cap())
	}

	v := Shape{1, 2}
	if !v.Equal(v) {
		t.Fatal("shape should be equal")
	}

	if v.Equal(Shape{1, 1}) {
		t.Fatal("shape should not be equal")
	}
}

func TestArray(t *testing.T) {
	arr := New(Shape{1, 3})
	t.Log(arr)
	t.Log(arr.data)
}

func TestArrayDot(t *testing.T) {
	arr1 := Init(Shape{1, 2}, 0.1)
	arr2 := Init(Shape{1, 2}, 0.5)

	val := arr1.Dot(arr2)
	if val != 0.1 {
		t.Fatal("Dot Array Fail")
	}
}
