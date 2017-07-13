package layer

import (
	"errors"
	"log"
	"math"

	"github.com/cvley/gocaffe/blob"
	pb "github.com/cvley/gocaffe/proto"
)

type SoftmaxLayer struct {
	axis   int
	bottom []string
	top    []string
	name   string
}

func NewSoftmaxLayer(param *pb.V1LayerParameter) (Layer, error) {
	softParam := param.GetSoftmaxParam()
	axis := -1
	if softParam != nil {
		axis = int(softParam.GetAxis())
	}

	return &SoftmaxLayer{
		axis:   axis,
		bottom: param.GetBottom(),
		top:    param.GetTop(),
		name:   param.GetName(),
	}, nil
}

func (soft *SoftmaxLayer) Forward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	top := bottom[0].Copy()

	// set up scale
	shape := bottom[0].Shape()
	if len(shape) != 4 {
		return nil, errors.New("not implement")
	}

	axis := soft.axis
	if soft.axis < 0 {
		axis += len(shape)
	}

	idx := make([]int, 4)
	for i := 0; i < len(shape); i++ {
		idx[i] = int(shape[i])
		if i > axis {
			idx[i] = 0
		}
	}

	log.Println("axis", axis)
	log.Println(top.DataString())

	var sum float64
	for i := axis; i < len(shape); i++ {
		for v := 0; v < int(shape[i]); v++ {
			idx[i] = v
			val := top.Get(idx)
			sum += math.Exp(val)
		}
	}

	top.Exp()
	top.Scale(1 / sum)

	return []*blob.Blob{top}, nil
}

func (soft *SoftmaxLayer) Type() string {
	return soft.name
}

func (soft *SoftmaxLayer) Bottom() []string {
	return soft.bottom
}

func (soft *SoftmaxLayer) Top() []string {
	return soft.top
}
