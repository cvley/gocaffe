package layer

import (
	"math"

	"github.com/cvley/gocaffe/blob"
	pb "github.com/cvley/gocaffe/proto"
)

// SigmoidLayer represents sigmoid function non-linearity y = (1 +
// \exp(-x))^{-1}, a classic choice in neural networks.
// Note that the gradient vanishes as the values move away from 0.
// The ReLULayer is often a better choice for this reason.
type SigmoidLayer struct {
	bottom []string
	top    []string
	name   string
}

func NewSigmoidLayer(param *pb.V1LayerParameter) (*SigmoidLayer, error) {
	return &SigmoidLayer{
		bottom: param.GetBottom(),
		top:    param.GetTop(),
		name:   param.GetName(),
	}, nil
}

func (s *SigmoidLayer) Forward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	top, err := blob.New(bottom[0].Shape())
	if err != nil {
		return nil, err
	}

	for n := 0; n < int(bottom[0].Num()); n++ {
		for c := 0; c < int(bottom[0].Channels()); c++ {
			for h := 0; h < int(bottom[0].Height()); h++ {
				for w := 0; w < int(bottom[0].Width()); w++ {
					idx := []int{n, c, h, w}
					v := bottom[0].Get(idx)
					top.Set(idx, sigmoid(v))
				}
			}
		}
	}

	return []*blob.Blob{top}, nil
}

func (s *SigmoidLayer) Type() string {
	return s.name
}

func (s *SigmoidLayer) Bottom() []string {
	return s.bottom
}

func (s *SigmoidLayer) Top() []string {
	return s.top
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
