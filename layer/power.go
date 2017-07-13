package layer

import (
	"errors"
	"math"

	"github.com/cvley/gocaffe/blob"
	pb "github.com/cvley/gocaffe/proto"
)

// PowerLayer computes y = (\alpha x + \beta) ^ \gamma as specified by the
// scale \alpha, shift \beta, and power \gamma
type PowerLayer struct {
	power     float64
	scale     float64
	shift     float64
	diffScale float64
}

func NewPowerLayer(param *pb.V1LayerParameter) (*PowerLayer, error) {
	powerParam := param.GetPowerParam()
	if powerParam == nil {
		return nil, errors.New("no power param")
	}

	power := float64(powerParam.GetPower())
	scale := float64(powerParam.GetScale())
	shift := float64(powerParam.GetShift())

	return &PowerLayer{
		power:     power,
		scale:     scale,
		shift:     shift,
		diffScale: power * scale,
	}, nil
}

// Forward compute y = (shift + scale * x) ^ power
func (p *PowerLayer) Forward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	// special case where we can ignore the input: scale or power is zero
	if p.diffScale == 0 {
		v := 1.0
		if p.power != 0 {
			v = math.Pow(p.shift, p.power)
		}
		top, err := blob.Init(bottom[0].Shape(), v)
		if err != nil {
			return nil, err
		}

		return []*blob.Blob{top}, nil
	}

	top := bottom[0].Copy()
	if p.scale != 1 {
		top.Scale(p.scale)
	}
	if p.shift != 0 {
		top.Shift(p.shift)
	}
	if p.power != 1 {
		top.Powx(p.power)
	}

	return []*blob.Blob{top}, nil
}

func (p *PowerLayer) String() string {
	return "PowerLayer"
}
