package layer

import (
	"math"

	"github.com/cvley/gocaffe/blob"
	pb "github.com/cvley/gocaffe/proto"
)

// ReLULayer represents Rectified Linear Unit non-linearity y = \max(0, x). The
// simple max is fast to compute
type ReLULayer struct {
	negative float64
}

func NewReLULayer(param *pb.V1LayerParameter) (Layer, error) {
	reluParam := param.GetReluParam()
	if reluParam == nil {
		return &ReLULayer{negative: 0}, nil
	}
	negative := reluParam.GetNegativeSlope()
	return &ReLULayer{negative: float64(negative)}, nil
}

func (relu *ReLULayer) Forward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	top, err := blob.New(bottom[0].Shape())
	if err != nil {
		return nil, err
	}

	for n := 0; n < bottom[0].Num(); n++ {
		for c := 0; c < bottom[0].Channels(); c++ {
			for h := 0; h < bottom[0].Height(); h++ {
				for w := 0; w < bottom[0].Width(); w++ {
					idx := []int{n, c, h, w}
					value := bottom[0].Get(idx, blob.ToData)
					reluV := math.Max(value, 0) + relu.negative*math.Min(value, 0)
					top.Set(idx, reluV, blob.ToData)
				}
			}
		}
	}

	return []*blob.Blob{top}, nil
}

func (relu *ReLULayer) Type() string {
	return "ReLU"
}
