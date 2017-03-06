package blob

import (
	"math"
	"testing"

	pb "github.com/cvley/gocaffe/proto"
	"github.com/golang/protobuf/proto"
)

func TestNewBlob(t *testing.T) {
	shape := []int{1, 1, 1, 1}
	b, err := New(shape)
	if err != nil {
		t.Fatal(err)
	}

	if len(b.shape) != len(shape) {
		t.Fatal("shape mismatch")
	}
	b.data = []float64{0.1}

	protobuf, err := b.ToProto(false)
	if err != nil {
		t.Fatal(err)
	}

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
	if err := proto.Unmarshal(protobuf, pbuf); err != nil {
		t.Fatal(err)
	}

	newBlob, err := FromProto(pbuf)
	if err != nil {
		t.Fatal(err)
	}

	sum := newBlob.L1Norm(ToData)
	if math.Abs(sum-0.1) > 1e-8 {
		t.Fatal("AsumData func fail")
	}

	b.Scale(0.1, ToData)
	if math.Abs(b.data[0]-0.01) > 1e-8 {
		t.Fatal("ScaleData func fail")
	}

	sqrSum := b.L2Norm(ToData)
	if math.Abs(sqrSum-0.0001) > 1e-8 {
		t.Fatal("SumSquareData func fail")
	}
}
