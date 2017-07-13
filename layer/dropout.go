package layer

import (
	"errors"

	"github.com/cvley/gocaffe/blob"
	pb "github.com/cvley/gocaffe/proto"
)

type DropoutLayer struct {
	threshold float64
	scale     float64
	bottom    []string
	top       []string
	name      string
}

func NewDropoutLayer(param *pb.V1LayerParameter) (Layer, error) {
	dropParam := param.GetDropoutParam()
	if dropParam == nil {
		return nil, errors.New("create dropout layer fail")
	}

	threshold := float64(dropParam.GetDropoutRatio())
	if threshold <= 0 || threshold >= 1 {
		return nil, errors.New("create dropout layer fail, invalid dropout ratio")
	}
	scale := 1 / (1 - threshold)

	return &DropoutLayer{
		threshold: threshold,
		scale:     scale,
		bottom:    param.GetBottom(),
		top:       param.GetTop(),
		name:      param.GetName(),
	}, nil
}

func (drop *DropoutLayer) Forward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	// training phase use bernoulli random number generator
	// test phase just return bottom
	return bottom, nil
}

func (drop *DropoutLayer) Bottom() []string {
	return drop.bottom
}

func (drop *DropoutLayer) Top() []string {
	return drop.top
}

func (drop *DropoutLayer) Type() string {
	return drop.name
}
