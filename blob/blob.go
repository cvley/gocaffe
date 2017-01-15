package blob

import (
	"bytes"
	"errors"
	"fmt"
	"math"

	pb "github.com/cvley/gocaffe/proto"
	"github.com/gonum/blas/blas64"
)

const kMaxBlobAxes = 32

type Blob struct {
	Data      []float64
	Diff      []float64
	ShapeData []float64
	Shape    []int32
	Count     int
	Capacity  int
}

func New() *Blob {
	return &Blob{
		Capacity: 0,
	}
}

func (b *Blob) FromProto(other *pb.BlobProto, reshape bool) error {
	// get shape
	if reshape {
		shape := []int32{}
		if other.GetHeight() != 0 || other.GetChannels() != 0 || other.GetNum() != 0 || other.GetWidth() != 0 {
			shape = append(shape, other.GetNum())
			shape = append(shape, other.GetChannels())
			shape = append(shape, other.GetHeight())
			shape = append(shape, other.GetWidth())
		} else {
			blobShape := other.GetShape()
			for _, v := range blobShape.GetDim() {
				shape = append(shape, int32(v))
			}
		}
		b.Reshape(shape)
	} else {
		if !b.ShapeEquals(other) {
			return errors.New("shape mismatch (reshape not set)")
		}
	}

	// copy data
	if len(other.GetData()) > 0 {
		if b.Count != len(other.GetData()) {
			return errors.New("get data fail: count mismatch data length")
		}
		b.Data = make([]float64, len(other.GetData()))
		for i, v := range other.GetData() {
			b.Data[i] = float64(v)
		}
	} else if len(other.GetDoubleData()) > 0 {
		if b.Count != len(other.GetDoubleData()) {
			return errors.New("get double data fail: count mismatch data length")
		}
		b.Data = other.GetDoubleData()
	}

	if len(other.GetDiff()) > 0 {
		if b.Count != len(other.GetDiff()) {
			return errors.New("get diff fail: count mismatch data length")
		}
		b.Diff = make([]float64, len(other.GetDiff()))
		for i, v := range other.GetDiff() {
			b.Diff[i] = float64(v)
		}
	} else if len(other.GetDoubleDiff()) > 0 {
		if b.Count != len(other.GetDoubleDiff()) {
			return errors.New("get double diff fail: count mismatch data length")
		}
		b.Diff = other.GetDoubleDiff()
	}

	return nil
}

func (b *Blob) ToProto(writeDiff bool) *pb.BlobProto {
	shape := []int64{}
	for _, k := range b.Shape {
		shape = append(shape, int64(k))
	}
	proto := &pb.BlobProto{
		Shape:      &pb.BlobShape{Dim: shape},
		DoubleData: b.Data,
	}

	if writeDiff {
		if b.Diff != nil {
			proto.DoubleDiff = b.Diff
		}
	}

	return proto
}

func (b *Blob) CanonicalAxisIndex(index int) int {
	if index > b.AxesNum() || index < -b.AxesNum() {
		panic("index is not in blob axes range")
	}

	if index < 0 {
		return index + b.AxesNum()
	}

	return index
}

func (b *Blob) ShapeEquals(other *pb.BlobProto) bool {
	if other.GetHeight() != 0 || other.GetChannels() != 0 || other.GetNum() != 0 || other.GetWidth() != 0 {
		return len(b.Shape) <= 4 && b.LegacyShape(-4) == other.GetNum() && b.LegacyShape(-3) == other.GetChannels() && b.LegacyShape(-2) == other.GetHeight() && b.LegacyShape(-1) == other.GetWidth()
	}

	otherShape := other.GetShape().GetDim()
	for i, v := range otherShape {
		if b.Shape[i] != int32(v) {
			return false
		}
	}
	return true
}

func (b *Blob) Reshape(shape []int32) {
	if len(shape) > kMaxBlobAxes {
		panic("blob shape dimensions larger than max blob axes")
	}

	b.Count = 1

	// reset size of shape and shapeData
	b.Shape = make([]int32, len(shape))
	if b.Shape != nil || len(b.Shape) < len(shape) {
		b.ShapeData = make([]float64, len(shape))
	}

	for i, v := range shape {
		if v < 0 {
			panic("shape value invalid")
		}
		if b.Count != 0 {
			if int(v) > math.MaxInt64/b.Count {
				panic("blob size exceeds MaxInt64")
			}
			b.Count *= int(v)
			b.Shape[i] = v
			b.ShapeData = append(b.ShapeData, float64(v))
		}
	}

	if b.Count > b.Capacity {
		b.Capacity = b.Count
		b.Data = make([]float64, b.Capacity)
		b.Diff = make([]float64, b.Capacity)
	}
}

func (b *Blob) ReshapeFromBlobShape(blobShape *pb.BlobShape) {
	if (len(blobShape.GetDim())) > kMaxBlobAxes {
		panic("blob shape dimensions larger than max blob axes")
	}

	shape := make([]int32, len(blobShape.GetDim()))
	for i, v := range blobShape.GetDim() {
		shape[i] = int32(v)
	}

	b.Reshape(shape)
}

func (b *Blob) ReshapeLike(other *Blob) {
	b.Reshape(other.Shape)
}

func (b *Blob) String() string {
	var buffers bytes.Buffer
	for _, v := range b.Shape {
		buffers.WriteString(fmt.Sprintf("%d ", v))
	}
	buffers.WriteString(fmt.Sprintf("(%d)", b.Count))

	return buffers.String()
}

func (b *Blob) ShapeOfIndex(index int) int32 {
	return b.Shape[index]
}

func (b *Blob) AxesNum() int {
	return len(b.Shape)
}

func (b *Blob) Num() int32 {
	return b.LegacyShape(0)
}

func (b *Blob) Channels() int32 {
	return b.LegacyShape(1)
}

func (b *Blob) Height() int32 {
	return b.LegacyShape(2)
}

func (b *Blob) Width() int32 {
	return b.LegacyShape(3)
}

func (b *Blob) LegacyShape(index int) int32 {
	if b.AxesNum() > 4 {
		panic("cannot use legacy accessors on Blobs with > 4 axes.")
	}
	if index > 4 && index < -4 {
		panic("index is not in [-4, 4]")
	}

	if index > b.AxesNum() || index < -b.AxesNum() {
		return 1
	}

	return b.Shape[index]
}

func (b *Blob) offset(indices []int32) int32 {
	if len(indices) > b.AxesNum() {
		panic("offset: indices larger than blob axes number")
	}

	offset := int32(0)
	for i := 0; i < b.AxesNum(); i++ {
		offset *= b.Shape[i]
		if len(indices) > i {
			if indices[i] > int32(0) && indices[i] < b.Shape[i] {
				offset += indices[i]
			}
		}
	}

	return offset
}

func (b *Blob) dataAt(index []int32) float64 {
	return b.Data[b.offset(index)]
}

func (b *Blob) diffAt(index []int32) float64 {
	return b.Diff[b.offset(index)]
}

// AsumData compute the sum of absolute values (L1 norm) of the data
func (b *Blob) AsumData() float64 {
	return blas64.Asum(b.Count, blas64.Vector{Data: b.Data})
}

// AsumDiff compute the sum of absolute values (L1 norm) of the diff
func (b *Blob) AsumDiff() float64 {
	return blas64.Asum(b.Count, blas64.Vector{Data: b.Diff})
}

// SumSquareData compute the sum of squares (L2 norm squared) of the data
func (b *Blob) SumSquareData() float64 {
	return blas64.Dot(b.Count, blas64.Vector{Data: b.Data}, blas64.Vector{Data: b.Data})
}

// SumSquareDiff compute the sum of squares (L2 norm squared) of the diff
func (b *Blob) SumSquareDiff() float64 {
	return blas64.Dot(b.Count, blas64.Vector{Data: b.Diff}, blas64.Vector{Data: b.Diff})
}

// ScaleData scale the blob data by a constant factor
func (b *Blob) ScaleData(scale float64) {
	blas64.Scal(b.Count, scale, blas64.Vector{Data: b.Data})
}

// ScaleDiff scale the blob diff by a constant factor
func (b *Blob) ScaleDiff(scale float64) {
	blas64.Scal(b.Count, scale, blas64.Vector{Data: b.Diff})
}
