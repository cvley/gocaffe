package layer

import (
	"errors"
	"log"

	"github.com/cvley/gocaffe/blob"
	pb "github.com/cvley/gocaffe/proto"
)

type InnerProductLayer struct {
	n         int
	biasTerm  bool
	weight    *blob.Blob
	bias      *blob.Blob
	transpose bool
	axis      int
	bottom    []string
	top       []string
	name      string
}

func NewInnerProductLayer(param *pb.V1LayerParameter) (*InnerProductLayer, error) {
	innerParam := param.GetInnerProductParam()
	if innerParam == nil {
		return nil, errors.New("create inner product layer fail, invalid param")
	}

	numOutput := innerParam.GetNumOutput()
	biasTerm := innerParam.GetBiasTerm()
	transpose := innerParam.GetTranspose()
	axis := innerParam.GetAxis()

	// TODO check if we need to initialize the weight and bias
	var weight, bias *blob.Blob
	blobprotos := param.GetBlobs()
	if blobprotos != nil {
		var err error
		weight, err = blob.FromProto(blobprotos[0])
		if err != nil {
			return nil, err
		}
		if biasTerm {
			bias, err = blob.FromProto(blobprotos[1])
			if err != nil {
				return nil, err
			}
		}
	}

	return &InnerProductLayer{
		biasTerm:  biasTerm,
		weight:    weight,
		bias:      bias,
		transpose: transpose,
		n:         int(numOutput),
		axis:      int(axis),
		bottom:    param.GetBottom(),
		top:       param.GetTop(),
		name:      param.GetName(),
	}, nil
}

func (inner *InnerProductLayer) Forward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	shape := bottom[0].Shape()
	log.Println("inner bottom shape", shape)

	reshape := make([]int64, len(shape))

	M := int64(1)
	for i := 0; i < inner.axis; i++ {
		M *= shape[i]
		reshape[i] = shape[i]
	}

	K := int64(1)
	for i := inner.axis; i < len(shape); i++ {
		K *= shape[i]
	}
	if K != inner.weight.Width() {
		log.Println("ERROR", K, inner.weight.Height())
		return nil, errors.New("Input size incompatible with inner product parameters.")
	}

	log.Println("inner weight and bias shape", inner.weight.Shape(), inner.bias.Shape())

	reBlob, err := bottom[0].Reshape([]int64{1, 1, M, K})
	if err != nil {
		return nil, err
	}

	log.Println("reshape bottom shape", reBlob.Shape())

	// top shape [1, 1, M, inner.n]
	top, err := reBlob.MMul(inner.weight.Trans())
	if err != nil {
		return nil, err
	}

	if inner.biasTerm {
		// bias shape [1, 1, 1, inner.n]
		biasMultiplier, err := blob.Init([]int64{1, 1, M, 1}, 1)
		if err != nil {
			return nil, err
		}
		mbias, err := biasMultiplier.MMul(inner.bias)
		if err != nil {
			return nil, err
		}
		if err := top.Add(mbias); err != nil {
			return nil, err
		}
	}

	if inner.Type() == "fc8" {
		log.Println("MMul top", top.DataString())
	}

	log.Println(inner.Type(), bottom[0].Shape(), "->", top.Shape())

	return []*blob.Blob{top}, nil
}

// Type of Layer
func (inner *InnerProductLayer) Type() string {
	return inner.name
}

func (inner *InnerProductLayer) Bottom() []string {
	return inner.bottom
}

func (inner *InnerProductLayer) Top() []string {
	return inner.top
}
