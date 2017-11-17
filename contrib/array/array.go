package array

import (
	"bytes"
	"fmt"
)

const (
	maxAxes = 32
)

type Shape []int

func (s Shape) isValid() bool {
	if len(s) > maxAxes {
		return false
	}

	for _, v := range s {
		if v <= 0 {
			return false
		}
	}

	return true
}

func (s Shape) Cap() int {
	cap := 1
	for _, v := range s {
		cap *= v
	}

	return cap
}

func (s Shape) Equal(x Shape) bool {
	if !s.isValid() || !x.isValid() {
		return false
	}

	if len(s) != len(x) {
		return false
	}

	for i, v := range s {
		if v != x[i] {
			return false
		}
	}

	return true
}

func (s Shape) String() string {
	buff := bytes.Buffer{}
	buff.Write([]byte("("))
	b := make([][]byte, len(s))
	for i, v := range s {
		b[i] = []byte(fmt.Sprintf("%d", v))
	}
	buff.Write(bytes.Join(b, []byte(",")))
	buff.Write([]byte(")"))
	return buff.String()
}

type Array struct {
	shape Shape
	data  []float64
}

func New(shape Shape) *Array {
	if !shape.isValid() {
		panic("invalid shape in new")
	}

	return &Array{
		shape: shape,
		data:  make([]float64, shape.Cap()),
	}
}

func Init(shape Shape, value float64) *Array {
	if !shape.isValid() {
		panic("invalid shape in init")
	}

	data := make([]float64, shape.Cap())
	for i, _ := range data {
		data[i] = value
	}

	return &Array{
		shape: shape,
		data:  data,
	}
}

func (arr *Array) Shape() Shape {
	return arr.shape
}

func (arr *Array) Copy() *Array {
	array := New(arr.shape)
	copy(array.data, arr.data)
	return array
}

func (arr *Array) Dot(array *Array) float64 {
	shape := arr.Shape()
	if !shape.Equal(array.Shape()) {
		panic("Dot two array should have same shape")
	}

	val := float64(0)
	for i, v := range arr.data {
		val += v * array.data[i]
	}

	return val
}

func (arr *Array) String() string {
	buff := bytes.Buffer{}
	buff.WriteString(arr.shape.String())

	return buff.String()
}
