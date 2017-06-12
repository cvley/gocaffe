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
			log.Println("inner product", bias.Shape())
		}
		log.Println("inner product", weight.Shape())
	}

	return &InnerProductLayer{
		biasTerm:  biasTerm,
		weight:    weight,
		bias:      bias,
		transpose: transpose,
		n:         int(numOutput),
		axis:      int(axis),
	}, nil
}

func (inner *InnerProductLayer) Forward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	shape := bottom[0].Shape()
	reshape := make([]int, len(shape))

	M := 1
	for i := 0; i < inner.axis; i++ {
		M *= shape[i]
		reshape[i] = shape[i]
	}

	K := 1
	for i := inner.axis; i < len(shape); i++ {
		K *= shape[i]
	}
	if K != inner.weight.Height() {
		return nil, errors.New("Input size incompatible with inner product parameters.")
	}
	reBlob, err := bottom[0].Reshape([]int{1, 1, K, M})
	if err != nil {
		return nil, err
	}

	// top shape [1, 1, M, inner.n]
	top, err := reBlob.MMul(inner.weight, blob.ToData)
	if err != nil {
		return nil, err
	}

	if inner.biasTerm {
		// bias shape [1, 1, 1, inner.n]
		biasMultiplier, err := blob.Init([]int{1, 1, M, 1}, 1, blob.ToData)
		if err != nil {
			return nil, err
		}
		mbias, err := biasMultiplier.MMul(inner.bias, blob.ToData)
		if err != nil {
			return nil, err
		}
		if err := top.Add(mbias, blob.ToData); err != nil {
			return nil, err
		}
	}

	return []*blob.Blob{top}, nil
}

// Type of Layer
func (inner *InnerProductLayer) Type() string {
	return "InnerProduct"
}
