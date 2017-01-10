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
	data      []float64
	diff      []float64
	shapeData []float64
	shape     []int
	count     int
	capacity  int
}

func New(shape []int) *Blob {
	blob := &Blob{
		capacity: 0,
	}

	blob.Reshape(shape)
	return blob
}

func (blob *Blob) FromProto(proto *pb.BlobProto, reshape bool) error {
	// get shape

	// copy data
}

func (blob *Blob) ToProto(proto *pb.BlobProto, writeDiff bool) error {

}

func (blob *Blob) Reshape(shape []int) {
	if len(shape) > kMaxBlobAxes {
		panic(fmt.Printf("blob shape dimensions %d larger than %d", len(shape), kMaxBlobAxes))
	}

	blob.count = 1

	// reset size of shape and shapeData
	blob.shape = make([]int, len(shape))
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
