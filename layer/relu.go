package layer

import (
	"log"
	"math"

	"github.com/cvley/gocaffe/blob"
	pb "github.com/cvley/gocaffe/proto"
)

// ReLULayer represents Rectified Linear Unit non-linearity y = \max(0, x). The
// simple max is fast to compute
type ReLULayer struct {
	negative float64
	bottom   []string
	top      []string
	name     string
}

func NewReLULayer(param *pb.V1LayerParameter) (Layer, error) {
	reluParam := param.GetReluParam()
	if reluParam == nil {
		return &ReLULayer{
			negative: 0,
			bottom:   param.GetBottom(),
			top:      param.GetTop(),
			name:     param.GetName(),
		}, nil
	}
	negative := reluParam.GetNegativeSlope()
	return &ReLULayer{
		negative: float64(negative),
		bottom:   param.GetBottom(),
		top:      param.GetTop(),
		name:     param.GetName(),
	}, nil
}

func (relu *ReLULayer) Forward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	top, err := blob.New(bottom[0].Shape())
	if err != nil {
		return nil, err
	}

	for n := 0; n < int(bottom[0].Num()); n++ {
		for c := 0; c < int(bottom[0].Channels()); c++ {
			for h := 0; h < int(bottom[0].Height()); h++ {
				for w := 0; w < int(bottom[0].Width()); w++ {
					idx := []int{n, c, h, w}
					value := bottom[0].Get(idx)
					reluV := math.Max(value, 0) + relu.negative*math.Min(value, 0)
					top.Set(idx, reluV)
				}
			}
		}
	}

	log.Println(relu.Type(), bottom[0].Shape(), "->", top.Shape())

	return []*blob.Blob{top}, nil
}

func (relu *ReLULayer) Type() string {
	return relu.name
}

func (relu *ReLULayer) Bottom() []string {
	return relu.bottom
}

func (relu *ReLULayer) Top() []string {
	return relu.top
}
