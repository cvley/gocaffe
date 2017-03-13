package layer

import (
	"github.com/cvley/gocaffe/blob"
	//pb "github.com/cvley/gocaffe/proto"
)

type SplitLayer struct {
	count int
}

func NewSplitLayer(count int) *SplitLayer {
	if count <= 0 {
		panic("new split layer fail, invalid count")
	}

	return &SplitLayer{count: count}
}

func (split *SplitLayer) Forward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	top := make([]*blob.Blob, split.count)

	for i := 0; i < split.count; i++ {
		top[i] = bottom[0].Copy()
	}

	return top, nil
}

// Type of Layer
func (split *SplitLayer) Type() string {
	return "SplitLayer"
}
