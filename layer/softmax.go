package layer

import (
	"log"
	"errors"
	"math"

	"github.com/cvley/gocaffe/blob"
	pb "github.com/cvley/gocaffe/proto"
)

type SoftmaxLayer struct {
	axis int
}

func NewSoftmaxLayer(param *pb.V1LayerParameter) (Layer, error) {
	softParam := param.GetSoftmaxParam()
	axis := -1
	if softParam != nil {
		axis = int(softParam.GetAxis())
	}

	log.Println("softmax axis", axis)

	return &SoftmaxLayer{axis: axis}, nil
}

func (soft *SoftmaxLayer) Forward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	top := bottom[0].Copy()

	// set up scale
	shape := bottom[0].Shape()
	axis := soft.axis
	if soft.axis < 0 {
		axis += len(shape)
	}
	shape[axis] = 1
	scale, err := blob.New(shape)
	if err != nil {
		return nil, err
	}

	if len(shape) != 4 {
		return nil, errors.New("not implement")
	}
	// init scale
	for n := 0; n < shape[0]; n++ {
		for c := 0; c < shape[1]; c++ {
			for h := 0; h < shape[2]; h++ {
				for w := 0; w < shape[3]; w++ {
					idx := []int{n, c, h, w}
					scale.Set(idx, top.Get(idx, blob.ToData), blob.ToData)
				}
			}
		}
	}

	// get scale max
	for n := 0; n < top.Num(); n++ {
		for c := 0; c < top.Channels(); c++ {
			for h := 0; h < top.Height(); h++ {
				for w := 0; w < top.Width(); w++ {
					idx := []int{n, c, h, w}
					sIdx := []int{n, c, h, w}
					sIdx[axis] = 0
					v1 := scale.Get(sIdx, blob.ToData)
					v2 := top.Get(idx, blob.ToData)
					scale.Set(sIdx, math.Max(v1, v2), blob.ToData)
				}
			}
		}
	}

	// subtract scale max and summary the data
	var sum float64
	for n := 0; n < top.Num(); n++ {
		for c := 0; c < top.Channels(); c++ {
			for h := 0; h < top.Height(); h++ {
				for w := 0; w < top.Width(); w++ {
					idx := []int{n, c, h, w}
					sIdx := []int{n, c, h, w}
					sIdx[axis] = 0
					v1 := scale.Get(sIdx, blob.ToData)
					v2 := top.Get(idx, blob.ToData)
					top.Set(idx, v2-v1, blob.ToData)
					sum += math.Exp(v2 - v1)
				}
			}
		}
	}

	top.Exp(blob.ToData)
	top.Scale(1/sum, blob.ToData)

	return []*blob.Blob{top}, nil
}

func (soft *SoftmaxLayer) Type() string {
	return "Softmax"
}
