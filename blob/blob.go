package blob

import (
	"bytes"
	"fmt"
	"github.com/cvley/gocaffe/proto"
	"math"
)

const kMaxBlobAxes = 32

type Blob struct {
	data      []float32
	diff      []float32
	shapeData []float32
	shape     []int
	count     int
	capacity  int
}

func New(num, channels, height, width int) *Blob {
}

func (blob *Blob) Update() {

}

func (blob *Blob) FromProto(proto *proto.BlobProto, reshape bool) error {
	if proto.

}

func (blob *Blob) ToProto(proto *proto.BlobProto, writeDiff bool) error {

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

func (blob *Blob) ReshapeFromBlobShape(blobShape *probo.BlobShape) {
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
