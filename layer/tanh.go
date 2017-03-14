package layer

import (
	"math"

	"github.com/cvley/gocaffe/blob"
	pb "github.com/cvley/gocaffe/proto"
)

type TanHLayer struct{}

func NewTanHLayer(param *pb.V1LayerParameter) (Layer, error) {
	return &TanHLayer{}, nil
}

func (*TanHLayer) Forward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	top, err := blob.New(bottom[0].Shape())
	if err != nil {
		return nil, err
	}

	for n := 0; n < bottom[0].Num(); n++ {
		for c := 0; c < bottom[0].Channels(); c++ {
			for h := 0; h < bottom[0].Height(); h++ {
				for w := 0; w < bottom[0].Width(); w++ {
					idx := []int{n, c, h, w}
					v := bottom[0].Get(idx, blob.ToData)
					top.Set(idx, tanH(v), blob.ToData)
				}
			}
		}
	}

	return []*blob.Blob{top}, nil
}

func (*TanHLayer) Type() string {
	return "TanH"
}

func tanH(x float64) float64 {
	return (1 - math.Exp(-2*x)) / (1 + math.Exp(-2*x))
}
