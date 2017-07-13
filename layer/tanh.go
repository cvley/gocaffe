package layer

import (
	"math"

	"github.com/cvley/gocaffe/blob"
	pb "github.com/cvley/gocaffe/proto"
)

type TanHLayer struct {
	bottom []string
	top    []string
	name string
}

func NewTanHLayer(param *pb.V1LayerParameter) (Layer, error) {
	return &TanHLayer{
		bottom: param.GetBottom(),
		top:    param.GetTop(),
		name: param.GetName(),
	}, nil
}

func (t *TanHLayer) Forward(bottom []*blob.Blob) ([]*blob.Blob, error) {
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
					top.Set(idx, tanH(v))
				}
			}
		}
	}

	return []*blob.Blob{top}, nil
}

func (t *TanHLayer) Type() string {
	return t.name
}

func (t *TanHLayer) Bottom() []string {
	return t.bottom
}

func (t *TanHLayer) Top() []string {
	return t.top
}

func tanH(x float64) float64 {
	return (1 - math.Exp(-2*x)) / (1 + math.Exp(-2*x))
}
