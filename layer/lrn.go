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
	scaleData, err := blob.Init(bottom[0].Shape(), lrn.k, blob.ToData)
	if err != nil {
		return nil, err
	}

	paddedSquare, err := blob.New([]int{
		bottom[0].Num(),
		bottom[0].Channels() + lrn.size - 1,
		bottom[0].Height(),
		bottom[0].Width(),
	})
	if err != nil {
		return nil, err
	}

	channels := bottom[0].Channels()
	width := bottom[0].Width()
	height := bottom[0].Height()

	alphaOverSize := lrn.alpha / float64(lrn.size)

	// go through the images
	for n := 0; n < bottom[0].Num(); n++ {
		// compute the padded square
		imBlob, err := bottom[0].Range(
			[]int{n, channels, height, width},
			[]int{n + 1, channels, height, width},
			blob.ToData,
		)
		if err != nil {
			return nil, err
		}

		sqrBlob, err := imBlob.Dot(imBlob, blob.ToData)
		if err != nil {
			return nil, err
		}
		paddedSquare.SetNumChannel(0, n+lrn.prePad, sqrBlob, blob.ToData)

		// create the first channel scale
		for c := 0; c < lrn.size; c++ {
			cBlob, err := paddedSquare.Range(
				[]int{n, c, height, width},
				[]int{n, c + 1, height, width},
				blob.ToData,
			)
			if err != nil {
				return nil, err
			}
			cBlob.Scale(alphaOverSize, blob.ToData)
			if err := scaleData.SetNumChannel(n, 0, cBlob, blob.ToData); err != nil {
				return nil, err
			}
		}

		for c := 1; c < channels; c++ {
			// copy previous scale
			cBlob, err := scaleData.Range(
				[]int{n, c - 1, height, width},
				[]int{n, c, height, width},
				blob.ToData,
			)
			if err := scaleData.SetNumChannel(n, c, cBlob, blob.ToData); err != nil {
				return nil, err
			}
			// add head
			hBlob, err := paddedSquare.Range(
				[]int{0, c + lrn.size - 1, height, width},
				[]int{0, c + lrn.size, height, width},
				blob.ToData,
			)
			if err != nil {
				return nil, err
			}
			hBlob.Scale(alphaOverSize, blob.ToData)
			if err := cBlob.Add(hBlob, blob.ToData); err != nil {
				return nil, err
			}
			// subtract tail
			tBlob, err := paddedSquare.Range(
				[]int{0, c - 1, height, width},
				[]int{0, c, height, width},
				blob.ToData,
			)
			tBlob.Scale(-alphaOverSize, blob.ToData)
			if err := cBlob.Add(tBlob, blob.ToData); err != nil {
				return nil, err
			}

			if err := scaleData.SetNumChannel(n, c, cBlob, blob.ToData); err != nil {
				return nil, err
			}
		}
	}

	// compute output
	scaleData.Powx(-lrn.beta, blob.ToData)
	top, err := bottom[0].Dot(scaleData, blob.ToData)
	if err != nil {
		return nil, err
	}

	return []*blob.Blob{top}, nil
}

func (lrn *LrnLayer) withinChannelForward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	return nil, nil
}
