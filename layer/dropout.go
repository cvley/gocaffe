package layer

import (
	"errors"

	"github.com/cvley/gocaffe/blob"
	pb "github.com/cvley/gocaffe/proto"
)

type DropoutLayer struct {
	threshold float64
	scale     float64
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
	}, nil
}

func (drop *DropoutLayer) Forward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	// training phase use bernoulli random number generator
	//randVec, err := blob.New(bottom[0].Shape())
	//if err != nil {
	//	return nil, err
	//}
	// test phase just return bottom
	return bottom, nil
}

func (drop *DropoutLayer) Type() string {
	return "Dropout"
}
