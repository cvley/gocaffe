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
	log.Println("soft", top.DataString())

	// set up scale
	shape := bottom[0].Shape()
	if len(shape) != 4 {
		return nil, errors.New("not implement")
	}

	var max float64
	for v := 0; v < int(shape[3]); v++ {
		idx := []int{1, 1, 1, v}
		val := top.Get(idx)
		if val > max {
			max = val
			log.Println(max, idx)
		}
	}

	var sum float64
	for v := 0; v < int(shape[3]); v++ {
		idx := []int{1, 1, 1, v}
		val := top.Get(idx)
		sum += math.Exp(val/max-1)
	}

	log.Println("Sum", sum, max)

	for v := 0; v < int(shape[3]); v++ {
		idx := []int{1, 1, 1, v}
		val := top.Get(idx)
		exp := math.Exp(val/max-1) / sum
		top.Set(idx, exp)
	}

	log.Println(top.DataString())

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
