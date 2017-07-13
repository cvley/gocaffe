package layer

import (
	"errors"
	"math"

	"github.com/cvley/gocaffe/blob"
	pb "github.com/cvley/gocaffe/proto"
)

type EltwiseLayer struct {
	op            pb.EltwiseParameter_EltwiseOp
	coeffs        []float64
	maxIdx        *blob.Blob
	stableProGrad bool
}

func NewEltwiseLayer(param *pb.V1LayerParameter) (*EltwiseLayer, error) {
	eltwiseParam := param.GetEltwiseParam()
	if eltwiseParam == nil {
		return nil, errors.New("create eltwise layer fail, invalid parameter")
	}

	coeff := eltwiseParam.GetCoeff()
	if coeff == nil {
		return nil, errors.New("create eltwise layer fail, invalid coeff")
	}

	coeffs := make([]float64, len(coeff))
	for i, v := range coeffs {
		coeffs[i] = float64(v)
	}

	return &EltwiseLayer{
		op:            eltwiseParam.GetOperation(),
		coeffs:        coeffs,
		stableProGrad: eltwiseParam.GetStableProdGrad(),
	}, nil
}

func (elt *EltwiseLayer) Forward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	switch elt.op {
	case pb.EltwiseParameter_PROD:
		top, err := bottom[0].Dot(bottom[1])
		if err != nil {
			return nil, err
		}
		for i := 2; i < len(bottom); i++ {
			top, err = top.Dot(bottom[i])
			if err != nil {
				return nil, err
			}
		}

		return []*blob.Blob{top}, nil

	case pb.EltwiseParameter_SUM:
		coeffs, err := blob.New(bottom[0].Shape())
		if err != nil {
			return nil, err
		}
		for n := 0; n < int(bottom[0].Num()); n++ {
			for c := 0; c < int(bottom[0].Channels()); c++ {
				for h := 0; h < int(bottom[0].Height()); h++ {
					for w := 0; w < int(bottom[0].Width()); w++ {
						index := coeffs.Offset([]int{n, c, h, w})
						coeffs.Set([]int{n, c, h, w}, elt.coeffs[index])
					}
				}
			}
		}
		top := make([]*blob.Blob, len(bottom))
		for i, v := range bottom {
			result, err := v.Dot(coeffs)
			if err != nil {
				return nil, err
			}
			top[i] = result
		}

		return top, nil

	case pb.EltwiseParameter_MAX:
		mask, err := blob.Init(bottom[0].Shape(), -1)
		if err != nil {
			return nil, err
		}
		top, err := blob.Init(bottom[0].Shape(), -math.MaxFloat64)
		if err != nil {
			return nil, err
		}
		for n := 0; n < int(bottom[0].Num()); n++ {
			for c := 0; c < int(bottom[0].Channels()); c++ {
				for h := 0; h < int(bottom[0].Height()); h++ {
					for w := 0; w < int(bottom[0].Width()); w++ {
						idxs := []int{n, c, h, w}
						data0 := bottom[0].Get(idxs)
						data1 := bottom[1].Get(idxs)
						if data0 > data1 {
							top.Set(idxs, data0)
							mask.Set(idxs, 0)
						} else {
							top.Set(idxs, data1)
							mask.Set(idxs, 1)
						}
					}
				}
			}
		}

		for i := 2; i < len(bottom); i++ {
			for n := 0; n < int(bottom[0].Num()); n++ {
				for c := 0; c < int(bottom[0].Channels()); c++ {
					for h := 0; h < int(bottom[0].Height()); h++ {
						for w := 0; w < int(bottom[0].Width()); w++ {
							idxs := []int{n, c, h, w}
							data0 := bottom[i].Get(idxs)
							data1 := top.Get(idxs)
							if data0 > data1 {
								top.Set(idxs, data0)
								mask.Set(idxs, float64(i))
							}
						}
					}
				}
			}
		}

		elt.maxIdx = mask
		return []*blob.Blob{top}, nil
	}

	return nil, errors.New("Unknown elementwise operation")
}

func (elt *EltwiseLayer) Type() string {
	return "Eltwise"
}
