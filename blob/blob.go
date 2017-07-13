package blob

import (
	"bytes"
	"errors"
	"fmt"
	"log"
	"math"
	"sort"

	pb "github.com/cvley/gocaffe/proto"
	"github.com/golang/protobuf/proto"
)

const maxBlobAxes = 32

var (
	// ErrInvalidShape indicates invalid shape array, i.e. a certern value is
	// less than or equal 0
	ErrInvalidShape = errors.New("invalid shape")
	// ErrExceedMaxAxes indicates exceed maximum axes error
	ErrExceedMaxAxes = errors.New("shape exceed maximum axes(32)")
)

// Blob is the basic data container in gocaffe
type Blob struct {
	data  []float64
	diff  []float64
	shape []int64
	cap   int64
}

// New returns Blob from input shape
func New(shape []int64) (*Blob, error) {
	if len(shape) > maxBlobAxes {
		return nil, ErrExceedMaxAxes
	}

	cap := int64(1)
	for _, v := range shape {
		if v <= 0 {
			return nil, ErrInvalidShape
		}
		cap *= v
	}
	return &Blob{
		data:  make([]float64, cap),
		diff:  make([]float64, cap),
		shape: shape,
		cap:   cap,
	}, nil
}

// Init returns Blob with input shape, initialise with input value and type
func Init(shape []int64, v float64) (*Blob, error) {
	b, err := New(shape)
	if err != nil {
		return nil, err
	}

	for i := 0; i < int(b.cap); i++ {
		b.data[i] = v
	}

	return b, nil
}

// FromProto returns Blob reconstruct from protobuf data
func FromProto(data *pb.BlobProto) (*Blob, error) {
	shape := []int64{}
	if data.GetHeight() != 0 || data.GetChannels() != 0 || data.GetNum() != 0 || data.GetWidth() != 0 {
		shape = append(shape, int64(data.GetNum()))
		shape = append(shape, int64(data.GetChannels()))
		shape = append(shape, int64(data.GetHeight()))
		shape = append(shape, int64(data.GetWidth()))
	} else {
		shape = data.GetShape().GetDim()
	}

	b, err := New(shape)
	if err != nil {
		return nil, err
	}

	// copy data
	if len(data.GetData()) > 0 {
		if int(b.cap) != len(data.GetData()) {
			return nil, errors.New("get data fail: count mismatch data length")
		}
		for i, v := range data.GetData() {
			b.data[i] = float64(v)
		}
	} else if len(data.GetDoubleData()) > 0 {
		if int(b.cap) != len(data.GetDoubleData()) {
			return nil, errors.New("get double data fail: count mismatch data length")
		}
		copy(b.data, data.GetDoubleData())
	}

	return b, nil
}

// ToProto return protobuf binary data of Blob
func (b *Blob) ToProto() ([]byte, error) {
	data := &pb.BlobProto{
		Shape:      &pb.BlobShape{Dim: b.shape},
		DoubleData: b.data,
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

// Copy returns a new blob with the same shape and data
func (b *Blob) Copy() *Blob {
	result, _ := New(b.shape)
	copy(result.data, b.data)
	return result
}

// Strings returns blob shape and capacity in string format
func (b *Blob) String() string {
	var buffers bytes.Buffer
	for _, v := range b.shape {
		buffers.WriteString(fmt.Sprintf("%d ", v))
	}
	buffers.WriteString(fmt.Sprintf("(%d)", b.cap))

	return buffers.String()
}

// Shape returns the shape of the blob
func (b *Blob) Shape() []int64 {
	return b.shape
}

// ShapeOfIndex returns the shape in the input index
func (b *Blob) ShapeOfIndex(index int) int64 {
	return b.shape[index]
}

// AxesNum returns the length of blob shape
func (b *Blob) AxesNum() int {
	return len(b.shape)
}

// Num returns number of legacy shape
func (b *Blob) Num() int64 {
	return b.LegacyShape(0)
}

// Channels returns channels of legacy shape
func (b *Blob) Channels() int64 {
	return b.LegacyShape(1)
}

// Height returns height of legacy shape
func (b *Blob) Height() int64 {
	return b.LegacyShape(2)
}

// Width returns width of legacy shape
func (b *Blob) Width() int64 {
	return b.LegacyShape(3)
}

// Capacity returns the capacity of blob
func (b *Blob) Capacity() int64 {
	return b.cap
}

// LegacyShape return index shape in the legacy
func (b *Blob) LegacyShape(index int) int64 {
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

	var offset int
	for i := 0; i < b.AxesNum(); i++ {
		offset *= int(b.shape[i])
		if len(indices) > i {
			if indices[i] > 0 && indices[i] < int(b.shape[i]) {
				offset += indices[i]
			}
		}
	}

	return offset
}

// Range returns a new Blob between two input indices, currently used for
// convolution
func (b *Blob) Range(indices1, indices2 []int) (*Blob, error) {
	if len(b.shape) != len(indices1) || len(b.shape) != len(indices2) ||
		len(b.shape) != 4 {
		return nil, errors.New("get range data fail, invalid indices")
	}

	shape := make([]int64, len(b.shape))
	for i, v := range indices1 {
		shape[i] = int64(indices2[i] - v)
		if shape[i] == 0 {
			shape[i] = 1
		}
	}

	result, err := New(shape)
	if err != nil {
		return nil, err
	}

	for n := indices1[0]; n < indices2[0]; n++ {
		for c := indices1[1]; c < indices2[1]; c++ {
			for h := indices1[2]; h < indices2[2]; h++ {
				for w := indices1[3]; w < indices2[3]; w++ {
					idx := []int{n, c, h, w}
					result.Set(idx, b.Get(idx))
				}
			}
		}
	}

	return result, nil
}

func (b *Blob) SetNumChannel(index0, index1 int, other *Blob) error {
	//	if b.Width() != other.Width() || b.Height() != other.Height() {
	//		return errors.New("set channel fail, mismatch shape")
	//	}

	if other.Num() != 1 || other.Channels() != 1 {
		return errors.New("set channel fail, invalid blob")
	}

	for h := 0; h < int(b.Height()); h++ {
		for w := 0; w < int(b.Width()); w++ {
			idx := []int{index0, index1, h, w}
			b.Set(idx, other.Get([]int{0, 0, h, w}))
		}
	}

	return nil
}

// Set will set value in the index with input type
func (b *Blob) Set(index []int, value float64) {
	b.data[b.Offset(index)] = value
}

// Get returns the value in the input index based on the type
func (b *Blob) Get(index []int) float64 {
	return b.data[b.Offset(index)]
}

// L1Norm compute the sum of absolute values (L1 norm) of the data
func (b *Blob) L1Norm() float64 {
	var sum float64
	for _, v := range b.data {
		sum += math.Abs(v)
	}

	return sum
}

// L2Norm compute the sum of squares (L2 norm squared) of the data
func (b *Blob) L2Norm() float64 {
	var sum float64
	for _, v := range b.data {
		sum += math.Pow(v, 2)
	}

	return sum
}

// Shift will shift the blob data  by the input value
func (b *Blob) Shift(shift float64) {
	for i, v := range b.data {
		b.data[i] = v + shift
	}
}

// Scale scale the blob data  by a constant factor
func (b *Blob) Scale(scale float64) {
	for i, v := range b.data {
		b.data[i] = v * scale
	}
}

// Add will add the data by a input blob
func (b *Blob) Add(other *Blob) error {
	//	if !b.ShapeEquals(other) {
	//		return errors.New("blob add data fail, mismatch shape")
	//	}

	for i := 0; i < int(other.cap); i++ {
		b.data[i] += other.data[i]
	}

	return nil
}

// Dot performs element-wise multiply data by a input blob
func (b *Blob) Dot(other *Blob) (*Blob, error) {
	//	if !b.ShapeEquals(other) {
	//		return nil, errors.New("blob add data fail, mismatch shape")
	//	}

	result, err := New(b.shape)
	if err != nil {
		return nil, err
	}

	for i := 0; i < int(b.cap); i++ {
		result.data[i] = b.data[i] * other.data[i]
	}

	return result, nil
}

// Mul perform matrix multiply data by a input blob
func (b *Blob) Mul(other *Blob) (float64, error) {
	//if !b.ShapeEquals(other) {
	//	return 0, errors.New("blob add data fail, mismatch shape")
	//}

	var sum float64
	for i := 0; i < int(b.cap); i++ {
		sum += b.data[i] * other.data[i]
	}

	return sum, nil
}

// Powx perform element-wise powx of the blob
func (b *Blob) Powx(x float64) {
	for i := 0; i < int(b.cap); i++ {
		b.data[i] = math.Pow(b.data[i], x)
	}
}

// Exp perform element-wise Exp function
func (b *Blob) Exp() {
	for i := 0; i < int(b.cap); i++ {
		b.data[i] = math.Exp(b.data[i])
	}
}

// Trans perform transpose of matrix
func (b *Blob) Trans() *Blob {
	nShape := b.shape
	nShape[2], nShape[3] = b.shape[3], b.shape[2]
	nb, err := New(nShape)
	if err != nil {
		panic(err)
	}

	for n := 0; n < int(nb.Num()); n++ {
		for c := 0; c < int(nb.Channels()); c++ {
			for h := 0; h < int(nb.Height()); h++ {
				for w := 0; w < int(nb.Width()); w++ {
					oIdx := []int{n, c, w, h}
					nIdx := []int{n, c, h, w}
					nb.Set(nIdx, b.Get(oIdx))
				}
			}
		}
	}

	return nb
}

// MMul performs matrix multiply
func (b *Blob) MMul(x *Blob) (*Blob, error) {
	if b.Width() != x.Height() {
		return nil, errors.New("blob matrix multiply fail, invalid shape")
	}

	shape := []int64{b.Num() * x.Num(), b.Channels() * x.Channels(), b.Height(), x.Width()}
	result, err := New(shape)
	if err != nil {
		return nil, err
	}

	log.Println(b.Num(), x.Num(), b.Channels(), x.Channels(), b.Height(), x.Width())

	for n1 := 0; n1 < int(b.Num()); n1++ {
		for n2 := 0; n2 < int(x.Num()); n2++ {
			for c1 := 0; c1 < int(b.Channels()); c1++ {
				for c2 := 0; c2 < int(x.Channels()); c2++ {
					for h := 0; h < int(b.Height()); h++ {
						for w := 0; w < int(x.Width()); w++ {
							row, _ := b.GetRow([]int{n1, c1}, h)
							col, _ := x.GetCol([]int{n2, c2}, w)
							v, err := row.Mul(col)
							if err != nil {
								return nil, err
							}
							idx := []int{n1*int(b.Num()) + n2, c1*int(b.Channels()) + c2, h, w}
							result.Set(idx, v)
						}
					}
				}
			}
		}
	}

	return result, nil
}

// GetCol returns a blob of shape 1x1x1xheight, used for MMul
func (b *Blob) GetCol(index []int, x int) (*Blob, error) {
	shape := []int64{1, 1, 1, b.Height()}
	result, err := New(shape)
	if err != nil {
		return nil, err
	}

	for i := 0; i < int(b.Height()); i++ {
		idx := []int{index[0], index[1], i, x}
		result.Set([]int{1, 1, 1, i}, b.Get(idx))
	}
	return result, nil
}

// GetRow returns a blob of shape 1x1x1xwidth, used for MMul
func (b *Blob) GetRow(index []int, x int) (*Blob, error) {
	shape := []int64{1, 1, 1, b.Width()}
	result, err := New(shape)
	if err != nil {
		return nil, err
	}

	for i := 0; i < int(b.Width()); i++ {
		idx := []int{index[0], index[1], x, i}
		result.Set([]int{1, 1, 1, i}, b.Get(idx))
	}
	return result, nil
}

// Reshape returns a blob of new shape
func (b *Blob) Reshape(index []int64) (*Blob, error) {
	count := int64(1)
	for _, v := range index {
		count *= v
	}
	if count != b.cap {
		return nil, errors.New("Reshape fail, invalid index")
	}

	result := b.Copy()
	result.shape = index
	return result, nil
}

func (b *Blob) DataString() string {
	var buffer bytes.Buffer
	buffer.WriteString(fmt.Sprintf("Shape %v\nData ", b.shape))
	for _, v := range b.data {
		buffer.WriteString(fmt.Sprintf("%f ", v))
	}
	return buffer.String()
}

// GetTop returns tops number indexes and probs
func (b *Blob) GetTop(num int) []Value {
	vals := []Value{}
	for i := 0; i < int(b.Width()); i++ {
		probs := b.Get([]int{1, 1, 1, i})
		v := Value{Index: i, Probs: probs}
		vals = append(vals, v)
	}

	sort.Sort(SortValue(vals))

	return vals[:num]
}

type Value struct {
	Index int
	Probs float64
}

type SortValue []Value

func (v SortValue) Len() int           { return len(v) }
func (v SortValue) Swap(i, j int)      { v[i], v[j] = v[j], v[i] }
func (v SortValue) Less(i, j int) bool { return v[i].Probs > v[j].Probs }
