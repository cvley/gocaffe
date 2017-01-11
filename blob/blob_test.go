package blob

import (
	"testing"

	pb "github.com/cvley/gocaffe/proto"
	"github.com/golang/protobuf/proto"
)

func TestNewBlob(t *testing.T) {
	shape := []int32{1, 1, 1, 1}
	b := New(shape)
	if len(b.shape) != len(shape) {
		t.Fatal("shape mismatch")
	}
	b.Data = []float64{0.1}

	protobuf := b.ToProto(false)
	data, err := proto.Marshal(protobuf)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%v", protobuf)

	pbuf := &pb.BlobProto{}
	if err := proto.Unmarshal(data, pbuf); err != nil {
		t.Fatal(err)
	}

	if err := b.FromProto(pbuf, true); err != nil {
		t.Fatal(err)
	}

	t.Logf("%+v", b)
}
