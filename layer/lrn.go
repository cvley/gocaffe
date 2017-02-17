package layer

import (
	"errors"
	"log"

	"github.com/cvley/gocaffe/blob"
	pb "github.com/cvley/gocaffe/proto"
)

type LrnLayer struct {
	Params *pb.LRNParameter

	size   int
	prePad int
	alpha  float64
	beta   float64
	k      float64
}

func NewLRNLayer(params *pb.V1LayerParameter) (Layer, error) {
	log.Println("construct LRN layer")

	param := params.GetLrnParam()
	if param == nil {
		return nil, errors.New("get LRN parameters fail")
	}

	size := param.GetLocalSize()
	if size%2 == 0 {
		return nil, errors.New("LRN only supports odd values for local size.")
	}

	prePad := (size - 1) / 2
	alpha := float64(param.GetAlpha())
	beta := float64(param.GetBeta())
	k := float64(param.GetK())
	return &LrnLayer{
		Params: param,
		size:   size,
		prePad: prePad,
		alpha:  alpha,
		beta:   beta,
		k:      k,
	}, nil
}

func (lrn *LrnLayer) Forward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	if bottom[0].AxesNum() != 4 {
		return nil, errors.New("Input must have 4 axes, corresponding to (num, channels, height, width)")
	}

	switch lrn.Params.GetNormRegion() {
	case pb.LRNParameter_ACROSS_CHANNELS:
		return lrn.crossChannelForward(bottom)

	case pb.LRNParameter_WITHIN_CHANNEL:
		return lrn.withinChannelForward(bottom)

	default:
		return nil, errors.New("Unknown normalization region.")
	}
}

func (lrn *LrnLayer) Type() string {
	return "LRN"
}

func (lrn *LrnLayer) crossChannelForward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	scaleData := blob.New()
	scaleData.ReshapeLike(bottom[0])
	// start with the constant value
	for i := 0; i < scaleData.Count; i++ {
		scaleData.Data[i] = lrn.k
	}

	paddedSquare := blob.New()
	paddedSquare.Reshape([]int{
		1,
		bottom[0].Channels() + lrn.size - 1,
		bottom[0].Height(),
		bottom[0].Width(),
	})

	alphaOverSize := lrn.alpha / lrn.size
	// go through the images
}

func (lrn *LrnLayer) withinChannelForward(bottom []*blob.Blob) ([]*blob.Blob, error) {
}
