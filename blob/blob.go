package blob

import (
	"bytes"
	"fmt"
	"log"
	"math"

	pb "github.com/cvley/gocaffe/proto"
)

const kMaxBlobAxes = 32

type Blob struct {
	DoubleData []float64
	DoubleDiff []float64
	FloatData  []float32
	FloatDiff  []float32
	shapeData  []float64
	shape      []int32
	count      int
	capacity   int
}

func New(shape []int32) *Blob {
	blob := &Blob{
		capacity: 0,
	}

	blob.Reshape(shape)
	return blob
}

func (blob *Blob) FromProto(other *pb.BlobProto, reshape bool) error {
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
		blob.Reshape(shape)
	} else {
		if !blob.ShapeEquals(other) {
			panic("shape mismatch (reshape not set)")
		}
	}

	// copy data
	if other.GetData() {
		if blob.count != len(other.GetData()) {
			panic("get data fail: count mismatch data length")
		}
		blob.FloatData = other.GetData()
	} else {
		if blob.count != len(other.GetDoubleData()) {
			panic("get double data fail: count mismatch data length")
		}
		blob.DoubleData = other.GetDoubleData()
	}

	if other.GetDiff() {
		if blob.count != len(other.GetDiff()) {
			panic("get diff fail: count mismatch data length")
		}
		blob.FloatDiff = other.GetDiff()
	} else {
		if blob.count != len(other.GetDoubleDiff()) {
			panic("get double diff fail: count mismatch data length")
		}
		blob.DoubleDiff = other.GetDoubleDiff()
	}
}

func (blob *Blob) ToProto(proto *pb.BlobProto, writeDiff bool) error {
	shape := []int64{}
	for _, k := range blob.shape {
		shape = append(shape, int64(k))
	}
	proto = &pb.BlobProto{
		Shape:      &pb.BlobShape{Dim: shape},
		Diff:       blob.FloatDiff,
		DoubleData: blob.DoubleData,
		DoubleDiff: blob.DoubleDiff,
	}
	if blob.FloatData != nil {
		proto.Data = blob.FloatData
	}
	if blob.DoubleData != nil {
		proto.DoubleData = blob.DoubleData
	}

	if writeDiff {
		if blob.FloatDiff != nil {
			proto.Diff = blob.FloatDiff
		}
		if blob.DoubleDiff != nil {
			proto.DoubleDiff = blob.DoubleDiff
		}
	}
}

func (blob *Blob) ShapeEquals(other *pb.BlobProto) bool {
	if other.GetHeight() != 0 || other.GetChannels() != 0 || other.GetNum() != 0 || other.GetWidth() != 0 {
		return len(blob.shape) <= 4 && blob.legacyShape(-4) == other.GetNum() && blob.legacyShape(-3) == other.GetChannels() && blob.legacyShape(-2) == other.GetHeight() && blob.legacyShape(-1) == other.GetWidth()
	}

	otherShape := other.GetShape().GetDim()
	for i, v := range otherShape {
		if blob.shape[i] != int32(v) {
			return false
		}
	}
	return true
}

func (blob *Blob) Reshape(shape []int32) {
	if len(shape) > kMaxBlobAxes {
		panic(fmt.Printf("blob shape dimensions %d larger than %d", len(shape), kMaxBlobAxes))
	}

	blob.count = 1

	// reset size of shape and shapeData
	blob.shape = make([]int32, len(shape))
	if blob.shape != nil || blob.shape.Size() < len(shape) {
		blob.shapeData.Reset(len(shape))
	}

	for i, v := range shape {
		if v < 0 {
			panic("shape value invalid")
		}
		if blob.count != 0 {
			if v > math.MaxInt64/blob.count {
				panic("blob size exceeds MaxInt64")
			}
			blob.count *= v
			blob.shape[i] = v
			blob.shapeData = v
		}
	}

	if blob.count > blob.capacity {
		blob.capacity = blob.count
		blob.data.Reset(blob.capacity)
		blob.diff.Reset(blob.capacity)
	}
}

func (blob *Blob) ReshapeFromBlobShape(blobShape *pb.BlobShape) {
	if (len(blobShape.GetDim())) > kMaxBlobAxes {
		panic(fmt.Printf("blob shape dimensions %d larger than %d", len(blobShape.GetDim(), kMaxBlobAxes)))
	}

	shape := make([]int, len(blobShape.GetDim()))
	for i, v := range blobShape.GetDim() {
		shape[i] = v
	}

	blob.Reshape(shape)
}

func (blob *Blob) ReshapeLike(other *Blob) {
	blob.Reshape(other.shape)
}

func (blob *Blob) String() string {
	var buffers bytes.Buffer
	for _, v := range blob.shape {
		buffers.WriteString(fmt.Sprintf("%d ", v))
	}
	buffers.WriteString(fmt.Sprintf("(%d)", blob.count))

	return buffers.String()
}

func (blob *Blob) Shape() []int {
	return blob.shape
}

func (blob *Blob) ShapeOfIndex(index int) int {
	return blob.shape[index]
}

func (blob *Blob) AxesNum() int {
	return len(blob.shape)
}

func (blob *Blob) Count() int {
	return blob.count
}

func (blob *Blob) num() int {
	return blob.legacyShape(0)
}

func (blob *Blob) channels() int {
	return blob.legacyShape(1)
}

func (blob *Blob) height() int {
	return blob.legacyShape(2)
}

func (blob *Blob) width() int {
	return blob.legacyShape(3)
}

func (blob *Blob) legacyShape(index int) int {
	if blob.AxesNum() > 4 {
		panic("cannot use legacy accessors on Blobs with > 4 axes.")
	}
	if index > 4 && index < -4 {
		panic("index is not in [-4, 4]")
	}

	if index > blob.AxesNum() || index < -blob.AxesNum() {
		return 1
	}

	return blob.shape[index]
}

func (blob *Blob) offset(indices []int) int {
	if len(indices) > blob.AxesNum() {
		panic("offset: indices larger than blob axes number")
	}

	offset := 0
	for i := 0; i < blob.AxesNum(); i++ {
		offset *= blob.shape[i]
		if len(indices) > i {
			if indices[i] > 0 && indices[i] < blob.shape[i] {
				offset += indices[i]
			}
		}
	}

	return offset
}

func (blob *Blob) dataAt(index []int) float64 {
	return blob.data[blob.offset(index)]
}

func (blob *Blob) diffAt(index []int) float64 {
	return blob.diff[blob.offset(index)]
}

// AsumData compute the sum of absolute values (L1 norm) of the data
func (blob *Blob) AsumData() float64 {
}

// AsumDiff compute the sum of absolute values (L1 norm) of the diff
func (blob *Blob) AsumDiff() float64 {
}

// SumSquareData compute the sum of squares (L2 norm squared) of the data
func (blob *Blob) SumSquareData() float64 {
}

// SumSquareDiff compute the sum of squares (L2 norm squared) of the diff
func (blob *Blob) SumSquareDiff() float64 {
}

// ScaleData scale the blob data by a constant factor
func (blob *Blob) ScaleData(scale float64) {
}

// ScaleDiff scale the blob diff by a constant factor
func (blob *Blob) ScaleDiff(scale float64) {
}
