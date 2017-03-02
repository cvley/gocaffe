package blob

import (
	"math"
	"testing"

	pb "github.com/cvley/gocaffe/proto"
	"github.com/golang/protobuf/proto"
)

func TestNewBlob(t *testing.T) {
	shape := []int{1, 1, 1, 1}
	b := New()
	b.Reshape(shape)
	if len(b.Shape) != len(shape) {
		t.Fatal("shape mismatch")
	}
	b.Data = []float64{0.1}

	protobuf := b.ToProto(false)
	data, err := proto.Marshal(protobuf)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%v", protobuf)

	if b.AxesNum() != 4 {
		t.Fatal("mismatch AxesNum")
	}

	if b.Num() != 1 {
		t.Fatal("mismatch Num")
	}

	if b.Channels() != 1 {
		t.Fatal("mismatch Channels")
	}

	if b.Height() != 1 {
		t.Fatal("mismatch Height")
	}

	if b.Width() != 1 {
		t.Fatal("mismatch Width")
	}

	pbuf := &pb.BlobProto{}
	if err := proto.Unmarshal(data, pbuf); err != nil {
		t.Fatal(err)
	}

	if err := b.FromProto(pbuf, true); err != nil {
		t.Fatal(err)
	}

	sum := b.AsumData()
	if math.Abs(sum-0.1) > 1e-8 {
		t.Fatal("AsumData func fail")
	}

	b.ScaleData(0.1)
	if math.Abs(b.Data[0]-0.01) > 1e-8 {
		t.Fatal("ScaleData func fail")
	}

	sqrSum := b.SumSquareData()
	if math.Abs(sqrSum-0.0001) > 1e-8 {
		t.Fatal("SumSquareData func fail")
	}
}
