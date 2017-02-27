package layer

import (
	"errors"
	"math"
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
		size:   int(size),
		prePad: int(prePad),
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
		bottom[0].Num(),
		bottom[0].Channels() + lrn.size - 1,
		bottom[0].Height(),
		bottom[0].Width(),
	})

	alphaOverSize := lrn.alpha / float64(lrn.size)
	// go through the images
	for n := 0; n < bottom[0].Num(); n++ {
		// compute the padded square
		sqrData := make([]float64, bottom[0].Capacity)
		for c := 0; c < bottom[0].Channels(); c++ {
			for h := 0; h < bottom[0].Height(); h++ {
				for w := 0; w < bottom[0].Width(); w++ {
					sqrData[bottom[0].Width()*bottom[0].Height()*c+bottom[0].Width()+w] = math.Pow(bottom[0].DataAt([]int{n, c, h, w}), 2)
				}
			}

		}
		paddedSquare.Data = sqrData
		// create the first channel scale
	}

	return nil, nil
}

func (lrn *LrnLayer) withinChannelForward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	return nil, nil
}
