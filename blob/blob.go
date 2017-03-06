package blob

import (
	"bytes"
	"errors"
	"fmt"
	"math"

	"github.com/golang/protobuf/proto"

	pb "github.com/cvley/gocaffe/proto"
)

const maxBlobAxes = 32

const (
	// ToData will get or update data in Blob
	ToData = iota
	// ToDiff will get or update diff in Blob
	ToDiff
)

var (
	// ErrInvalidShape indicates invalid shape array, i.e. a certern value is
	// less than or equal 0
	ErrInvalidShape = errors.New("invalid shape")
	// ErrExceedMaxAxes indicates exceed maximum axes error
	ErrExceedMaxAxes = errors.New("shape exceed maximum axes(32)")
)

// Blob is the basic container in gocaffe
type Blob struct {
	data     []float64
	diff     []float64
	shape    []int
	capacity int
}

// New returns Blob from input shape
func New(shape []int) (*Blob, error) {
	if len(shape) > maxBlobAxes {
		return nil, ErrExceedMaxAxes
	}

	cap := 1
	for _, v := range shape {
		if v <= 0 {
			return nil, ErrInvalidShape
		}
		cap *= v
	}
	return &Blob{
		data:     make([]float64, cap),
		diff:     make([]float64, cap),
		shape:    shape,
		capacity: cap,
	}, nil
}

// Init returns Blob with input shape, initialise with input value and type
func Init(shape []int, v float64, tp int) (*Blob, error) {
	b, err := New(shape)
	if err != nil {
		return nil, err
	}

	switch tp {
	case ToData:
		for i, v := range b.data {
			b.data[i] = v
		}

	case ToDiff:
		for i, v := range b.diff {
			b.diff[i] = v
		}

	default:
		return nil, errors.New("initialise type not supported")
	}

	return b, nil
}

// FromProto returns Blob reconstruct from protobuf data
func FromProto(data *pb.BlobProto) (*Blob, error) {
	shape := []int{}
	if data.GetHeight() != 0 || data.GetChannels() != 0 || data.GetNum() != 0 || data.GetWidth() != 0 {
		shape = append(shape, int(data.GetNum()))
		shape = append(shape, int(data.GetChannels()))
		shape = append(shape, int(data.GetHeight()))
		shape = append(shape, int(data.GetWidth()))
	} else {
		blobShape := data.GetShape()
		for _, v := range blobShape.GetDim() {
			shape = append(shape, int(v))
		}
	}

	b, err := New(shape)
	if err != nil {
		return nil, err
	}

	// copy data
	if len(data.GetData()) > 0 {
		if b.capacity != len(data.GetData()) {
			return nil, errors.New("get data fail: count mismatch data length")
		}
		for i, v := range data.GetData() {
			b.data[i] = float64(v)
		}
	} else if len(data.GetDoubleData()) > 0 {
		if b.capacity != len(data.GetDoubleData()) {
			return nil, errors.New("get double data fail: count mismatch data length")
		}
		copy(b.data, data.GetDoubleData())
	}

	if len(data.GetDiff()) > 0 {
		if b.capacity != len(data.GetDiff()) {
			return nil, errors.New("get diff fail: count mismatch data length")
		}
		for i, v := range data.GetDiff() {
			b.diff[i] = float64(v)
		}
	} else if len(data.GetDoubleDiff()) > 0 {
		if b.capacity != len(data.GetDoubleDiff()) {
			return nil, errors.New("get double diff fail: count mismatch data length")
		}
		copy(b.diff, data.GetDoubleDiff())
	}

	return b, nil
}

// ToProto return protobuf binary data of Blob
func (b *Blob) ToProto(writeDiff bool) ([]byte, error) {
	shape := make([]int64, len(b.shape))
	for i, k := range b.shape {
		shape[i] = int64(k)
	}
	data := &pb.BlobProto{
		Shape:      &pb.BlobShape{Dim: shape},
		DoubleData: b.data,
	}

	if writeDiff {
		data.DoubleDiff = b.diff
	}

	return proto.Marshal(data)
}

// ShapeEquals returns whether two blob have the same shape
func (b *Blob) ShapeEquals(other *Blob) bool {
	for i, v := range b.shape {
		if v != other.shape[i] {
			return false
		}
	}

	return true
}

// Strings returns blob shape and capacity in string format
func (b *Blob) String() string {
	var buffers bytes.Buffer
	for _, v := range b.shape {
		buffers.WriteString(fmt.Sprintf("%d ", v))
	}
	buffers.WriteString(fmt.Sprintf("(%d)", b.capacity))

	return buffers.String()
}

// ShapeOfIndex returns the shape in the input index
func (b *Blob) ShapeOfIndex(index int) int {
	return b.shape[index]
}

// AxesNum returns the length of blob shape
func (b *Blob) AxesNum() int {
	return len(b.shape)
}

// Num returns number of legacy shape
func (b *Blob) Num() int {
	return b.LegacyShape(0)
}

// Channels returns channels of legacy shape
func (b *Blob) Channels() int {
	return b.LegacyShape(1)
}

// Height returns height of legacy shape
func (b *Blob) Height() int {
	return b.LegacyShape(2)
}

// Width returns width of legacy shape
func (b *Blob) Width() int {
	return b.LegacyShape(3)
}

// LegacyShape return index shape in the legacy
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

	return b.shape[index]
}

// Offset returns data offset of input indices
func (b *Blob) Offset(indices []int) int {
	if len(indices) > b.AxesNum() {
		panic("offset: indices larger than blob axes number")
	}

	offset := 1
	for i := 0; i < b.AxesNum(); i++ {
		offset *= b.shape[i]
		if len(indices) > i {
			if indices[i] > 0 && indices[i] < b.shape[i] {
				offset += indices[i]
			}
		}
	}

	return offset
}

// Range returns a new Blob between two input indices
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

	blob, err := New(shape)
	if err != nil {
		return nil, err
	}

	for i := offset1; i < offset2; i++ {
		blob.data[i-offset1] = b.data[i]
		blob.diff[i-offset1] = b.diff[i]
	}

	return blob, nil
}

// Set will set value in the index with input type
func (b *Blob) Set(index []int, value float64, tp int) {
	switch tp {
	case ToData:
		b.data[b.Offset(index)] = value

	case ToDiff:
		b.diff[b.Offset(index)] = value

	default:
		panic("Set Blob fail, invalid type")
	}
}

// Get returns the value in the input index based on the type
func (b *Blob) Get(index []int, tp int) float64 {
	switch tp {
	case ToData:
		return b.data[b.Offset(index)]

	case ToDiff:
		return b.diff[b.Offset(index)]
	}

	return math.MaxFloat64
}

// L1Norm compute the sum of absolute values (L1 norm) of the data or diff
func (b *Blob) L1Norm(tp int) float64 {
	var sum float64
	switch tp {
	case ToData:
		for _, v := range b.data {
			sum += math.Abs(v)
		}

	case ToDiff:
		for _, v := range b.diff {
			sum += math.Abs(v)
		}
	}

	return sum
}

// L2Norm compute the sum of squares (L2 norm squared) of the data or diff
func (b *Blob) L2Norm(tp int) float64 {
	var sum float64
	switch tp {
	case ToData:
		for _, v := range b.data {
			sum += math.Pow(v, 2)
		}

	case ToDiff:
		for _, v := range b.diff {
			sum += math.Pow(v, 2)
		}

	}

	return sum
}

// Scale scale the blob data or diff by a constant factor
func (b *Blob) Scale(scale float64, tp int) {
	switch tp {
	case ToData:
		for i, v := range b.data {
			b.data[i] = v * scale
		}

	case ToDiff:
		for i, v := range b.diff {
			b.diff[i] = v * scale
		}
	}
}

// Add will add the data or diff by a input blob
func (b *Blob) Add(other *Blob, tp int) error {
	if !b.ShapeEquals(other) {
		return errors.New("blob add data fail, mismatch shape")
	}

	switch tp {
	case ToData:
		for i := 0; i < b.capacity; i++ {
			b.data[i] += other.data[i]
		}
	case ToDiff:
		for i := 0; i < b.capacity; i++ {
			b.diff[i] += other.diff[i]
		}
	}

	return nil
}

// Mul will multiply data or diff by a input blob
func (b *Blob) Mul(other *Blob, tp int) error {
	if !b.ShapeEquals(other) {
		return errors.New("blob add data fail, mismatch shape")
	}

	switch tp {
	case ToData:
		for i := 0; i < b.capacity; i++ {
			b.data[i] *= other.data[i]
		}
	case ToDiff:
		for i := 0; i < b.capacity; i++ {
			b.diff[i] *= other.diff[i]
		}
	}

	return nil
}
