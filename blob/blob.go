package blob

import (
	"bytes"
	"errors"
	"fmt"
	"math"

	pb "github.com/cvley/gocaffe/proto"
)

const kMaxBlobAxes = 32

var (
	ErrEmptyBlob = errors.New("blob is empty")
)

type Blob struct {
	Data     []float64
	Diff     []float64
	Shape    []int
	Capacity int
}

func New() *Blob {
	return &Blob{
		Capacity: 0,
	}
}

func (b *Blob) InitData(v float64) error {
	if b.Capacity == 0 {
		return ErrEmptyBlob
	}

	for i := 0; i < b.Capacity; i++ {
		b.Data[i] = v
	}

	return nil
}

func (b *Blob) InitDiff(v float64) error {
	if b.Capacity == 0 {
		return ErrEmptyBlob
	}

	for i := 0; i < b.Capacity; i++ {
		b.Diff[i] = v
	}

	return nil
}

func (b *Blob) FromProto(other *pb.BlobProto, reshape bool) error {
	// get shape
	if reshape {
		shape := []int{}
		if other.GetHeight() != 0 || other.GetChannels() != 0 || other.GetNum() != 0 || other.GetWidth() != 0 {
			shape = append(shape, int(other.GetNum()))
			shape = append(shape, int(other.GetChannels()))
			shape = append(shape, int(other.GetHeight()))
			shape = append(shape, int(other.GetWidth()))
		} else {
			blobShape := other.GetShape()
			for _, v := range blobShape.GetDim() {
				shape = append(shape, int(v))
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
		if b.Capacity != len(other.GetData()) {
			return errors.New("get data fail: count mismatch data length")
		}
		b.Data = make([]float64, len(other.GetData()))
		for i, v := range other.GetData() {
			b.Data[i] = float64(v)
		}
	} else if len(other.GetDoubleData()) > 0 {
		if b.Capacity != len(other.GetDoubleData()) {
			return errors.New("get double data fail: count mismatch data length")
		}
		copy(b.Data, other.GetDoubleData())
	}

	if len(other.GetDiff()) > 0 {
		if b.Capacity != len(other.GetDiff()) {
			return errors.New("get diff fail: count mismatch data length")
		}
		b.Diff = make([]float64, len(other.GetDiff()))
		for i, v := range other.GetDiff() {
			b.Diff[i] = float64(v)
		}
	} else if len(other.GetDoubleDiff()) > 0 {
		if b.Capacity != len(other.GetDoubleDiff()) {
			return errors.New("get double diff fail: count mismatch data length")
		}
		copy(b.Diff, other.GetDoubleDiff())
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
		if len(b.Diff) != 0 {
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

func (b *Blob) ShapeEquals(other *Blob) bool {
	return other.Num() != b.Num() && other.AxesNum() != b.AxesNum() &&
		other.Channels() != b.Channels() && other.Capacity != b.Capacity
}

func (b *Blob) Reshape(shape []int) error {
	if len(shape) > kMaxBlobAxes {
		panic("blob shape dimensions larger than max blob axes")
	}

	count := 1
	for _, v := range shape {
		if v < 0 {
			return errors.New("invalid shape value")
		}
		count *= v
	}

	// reset size of shape
	b.Shape = make([]int, len(shape))
	copy(b.Shape, shape)

	if count > b.Capacity {
		b.Capacity = count
		b.Data = make([]float64, count)
		b.Diff = make([]float64, count)
	}

	return nil
}

func (b *Blob) ReshapeFromBlobShape(blobShape *pb.BlobShape) {
	if (len(blobShape.GetDim())) > kMaxBlobAxes {
		panic("blob shape dimensions larger than max blob axes")
	}

	shape := make([]int, len(blobShape.GetDim()))
	for i, v := range blobShape.GetDim() {
		shape[i] = int(v)
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
	buffers.WriteString(fmt.Sprintf("(%d)", b.Capacity))

	return buffers.String()
}

func (b *Blob) ShapeOfIndex(index int) int {
	return b.Shape[index]
}

func (b *Blob) AxesNum() int {
	return len(b.Shape)
}

func (b *Blob) Num() int {
	return b.LegacyShape(0)
}

func (b *Blob) Channels() int {
	return b.LegacyShape(1)
}

func (b *Blob) Height() int {
	return b.LegacyShape(2)
}

func (b *Blob) Width() int {
	return b.LegacyShape(3)
}

func (b *Blob) LegacyShape(index int) int {
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

func (b *Blob) Offset(indices []int) int {
	if len(indices) > b.AxesNum() {
		panic("offset: indices larger than blob axes number")
	}

	offset := 1
	for i := 0; i < b.AxesNum(); i++ {
		offset *= b.Shape[i]
		if len(indices) > i {
			if indices[i] > 0 && indices[i] < b.Shape[i] {
				offset += indices[i]
			}
		}
	}

	return offset
}

func (b *Blob) Range(indices1, indices2 []int) (*Blob, error) {
	offset1 := b.Offset(indices1)
	offset2 := b.Offset(indices2)

	if offset1 >= offset2 {
		return nil, errors.New("get range data fail, invalid indices")
	}

	shape := make([]int, len(indices1))
	for i, idx := range indices1 {
		if idx == indices2[i] {
			shape[i] = idx
		}
		if idx < indices2[i] {
			shape[i] = indices2[i] - idx
		}
	}

	blob := New()
	if err := blob.Reshape(shape); err != nil {
		return nil, err
	}

	for i := offset1; i < offset2; i++ {
		blob.Data[i-offset1] = b.Data[i]
		blob.Diff[i-offset1] = b.Diff[i]
	}

	return blob, nil
}

func (b *Blob) DataSet(index []int, value float64) {
	b.Data[b.Offset(index)] = value
}

func (b *Blob) DiffSet(index []int, value float64) {
	b.Diff[b.Offset(index)] = value
}

func (b *Blob) DataAt(index []int) float64 {
	return b.Data[b.Offset(index)]
}

func (b *Blob) DiffAt(index []int) float64 {
	return b.Diff[b.Offset(index)]
}

// AsumData compute the sum of absolute values (L1 norm) of the data
func (b *Blob) AsumData() float64 {
	var sum float64
	for _, v := range b.Data {
		sum += math.Abs(v)
	}
	return sum
}

// AsumDiff compute the sum of absolute values (L1 norm) of the diff
func (b *Blob) AsumDiff() float64 {
	var sum float64
	for _, v := range b.Diff {
		sum += math.Abs(v)
	}
	return sum
}

// SumSquareData compute the sum of squares (L2 norm squared) of the data
func (b *Blob) SumSquareData() float64 {
	var sum float64
	for _, v := range b.Data {
		sum += v * v
	}
	return sum
}

// SumSquareDiff compute the sum of squares (L2 norm squared) of the diff
func (b *Blob) SumSquareDiff() float64 {
	var sum float64
	for _, v := range b.Diff {
		sum += v * v
	}
	return sum
}

// Scale scale the blob data by a constant factor
func (b *Blob) ScaleData(scale float64, toDiff bool) {
	data := make([]float64, len(b.Data))
	for i, v := range b.Data {
		data[i] = v * scale
	}
	copy(b.Data, data)

	if toDiff {
		diff := make([]float64, len(b.Diff))
		for i, v := range b.Diff {
			diff[i] = v * scale
		}
		copy(b.Diff, diff)
	}
}

func (b *Blob) Add(blob *Blob, toDiff bool) error {
	if !b.ShapeEquals(blob) {
		return errors.New("blob add data fail, mismatch shape")
	}

	for i := 0; i < b.Capacity; i++ {
		b.Data[i] += blob.Data[i]
	}

	if toDiff {
		for i := 0; i < b.Capacity; i++ {
			b.Diff[i] += blob.Diff[i]
		}
	}

	return nil
}

func (b *Blob) Mul(blob *Blob, toDiff bool) error {
	if !b.ShapeEquals(blob) {
		return errors.New("blob add data fail, mismatch shape")
	}

	for i := 0; i < b.Capacity; i++ {
		b.Data[i] *= blob.Data[i]
	}

	if toDiff {
		for i := 0; i < b.Capacity; i++ {
			b.Diff[i] *= blob.Diff[i]
		}
	}

	return nil
}
