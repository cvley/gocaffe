package layer

import (
	"errors"
	"log"
	"math"

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
	bottom []string
	top    []string
	name   string
}

func NewLRNLayer(params *pb.V1LayerParameter) (Layer, error) {
	param := params.GetLrnParam()
	if param == nil {
		return nil, errors.New("get LRN parameters fail")
	}

	name := params.GetName()
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
		bottom: params.GetBottom(),
		top:    params.GetTop(),
		name:   name,
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
	return lrn.name
}

func (lrn *LrnLayer) Bottom() []string {
	return lrn.bottom
}

func (lrn *LrnLayer) Top() []string {
	return lrn.top
}

func (lrn *LrnLayer) crossChannelForward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	log.Println("LRN cross channel normalise")
	log.Println(lrn)

	top := bottom[0].Copy()
	for n := 0; n < int(bottom[0].Num()); n++ {
		for h := 0; h < int(bottom[0].Height()); h++ {
			for w := 0; w < int(bottom[0].Width()); w++ {
				for c := 0; c < int(bottom[0].Channels()); c++ {
					var sum float64
					for i := -lrn.prePad; i <= lrn.prePad; i++ {
						if (c+i) < 0 && (c+i) >= int(bottom[0].Channels()) {
							continue
						}
						idx := []int{n, c + i, h, w}
						sum += math.Pow(bottom[0].Get(idx), 2)
					}
					v := 1 + lrn.alpha/float64(lrn.size)*sum
					top.Set([]int{n, c, h, w}, math.Pow(v, lrn.beta))
				}
			}
		}
	}

	log.Println(lrn.Type(), bottom[0].Shape(), "->", top.Shape())

	return []*blob.Blob{top}, nil
}

func (lrn *LrnLayer) withinChannelForward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	// set up split layer of two output: one for product input, another for
	// square input
	splitLayer := NewSplitLayer(2)
	splitBlobs, _ := splitLayer.Forward(bottom)

	// set up square layer to square the input
	power := float32(2.0)
	squareParam := &pb.PowerParameter{Power: &power}
	powerLayer, err := NewPowerLayer(&pb.V1LayerParameter{PowerParam: squareParam})
	if err != nil {
		return nil, err
	}
	squareOutput, err := powerLayer.Forward([]*blob.Blob{splitBlobs[0]})
	if err != nil {
		return nil, err
	}

	// set up pool layer to sum over square neighborhoods of the input
	pool := pb.PoolingParameter_AVE
	prePad := uint32(lrn.prePad)
	kernelSize := uint32(lrn.size)
	poolParam := &pb.PoolingParameter{
		Pool:       &pool,
		Pad:        &prePad,
		KernelSize: &kernelSize,
	}
	poolLayer, err := NewPoolingLayer(&pb.V1LayerParameter{PoolingParam: poolParam})
	if err != nil {
		return nil, err
	}
	poolOutput, err := poolLayer.Forward(squareOutput)
	if err != nil {
		return nil, err
	}

	// set up power layer to compute (1 + alpha / N^2 s) ^ -beta, where s is
	// the sum of a squared neighborhood (the output of pool layer)
	beta := float32(-lrn.beta)
	alpha := float32(lrn.alpha)
	shift := float32(1)
	powerParam := &pb.PowerParameter{
		Power: &beta,
		Scale: &alpha,
		Shift: &shift,
	}

	powerLayer, err = NewPowerLayer(&pb.V1LayerParameter{PowerParam: powerParam})
	if err != nil {
		return nil, err
	}
	powerOutput, err := powerLayer.Forward(poolOutput)
	if err != nil {
		return nil, err
	}

	// set up a product layer to compute outputs by multiplying inputs by the
	// inverse demoninator computed by the power layer
	op := pb.EltwiseParameter_PROD
	productParam := &pb.EltwiseParameter{
		Operation: &op,
	}
	productLayer, err := NewEltwiseLayer(&pb.V1LayerParameter{EltwiseParam: productParam})
	if err != nil {
		return nil, err
	}

	productInput := []*blob.Blob{splitBlobs[1]}
	productInput = append(productInput, powerOutput...)
	top, err := productLayer.Forward(productInput)
	if err != nil {
		return nil, err
	}

	log.Println(lrn.Type(), bottom[0].Shape(), "->", top[0].Shape())
	return top, nil
}
