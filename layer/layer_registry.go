package layer

import (
	"fmt"
	"log"
	"strings"

	"github.com/cvley/gocaffe/blob"
	pb "github.com/cvley/gocaffe/proto"
)

var (
	LayerRegister LayerRegistry
)

type Layer interface {
	Forward(bottom []*blob.Blob) ([]*blob.Blob, error)
	Type() string
}

// NeuronLayer is the interface for layers that take one blob as input and
// produce one equally-sized blob as output, where each element of the output
// depends only on the corresponding input element
type NeuronLayer interface {
	Forward(bottom []*blob.Blob) ([]*blob.Blob, error)
	Type() string
}

type Creator func(*pb.V1LayerParameter) (Layer, error)

type LayerRegistry map[string]Creator

func init() {
	LayerRegister = make(LayerRegistry)
	LayerRegister.AddCreator("CONVOLUTION", GetConvolutionLayer)
	LayerRegister.AddCreator("RELU", GetReLULayer)
	LayerRegister.AddCreator("POOLING", GetPoolLayer)
	LayerRegister.AddCreator("LRN", GetLRNLayer)
	LayerRegister.AddCreator("INNER_PRODUCT", GetInnerProductLayer)
	LayerRegister.AddCreator("DROPOUT", GetDropoutLayer)
	LayerRegister.AddCreator("SOFTMAX", GetSoftmaxLayer)

	LayerRegister.AddCreator("Sigmoid", GetSigmoidLayer)
	LayerRegister.AddCreator("TanH", GetTanHLayer)
}

func (r LayerRegistry) AddCreator(tp string, creator Creator) error {
	if r.layerExist(tp) {
		return fmt.Errorf("Layer type %s already registered.")
	}
	r[tp] = creator
	return nil
}

func (r LayerRegistry) CreateLayer(param *pb.V1LayerParameter) (Layer, error) {
	tp := param.GetType().String()
	if !r.layerExist(tp) {
		return nil, fmt.Errorf("layer %s not exist", tp)
	}

	log.Printf("Creating layer %s", param.GetName())
	return r[tp](param)
}

func (r LayerRegistry) LayerTypeList() []string {
	result := []string{}
	for name, _ := range r {
		result = append(result, name)
	}
	return result
}

func (r LayerRegistry) LayerTypeListString() string {
	typeList := r.LayerTypeList()
	return strings.Join(typeList, ", ")
}

func (r LayerRegistry) layerExist(name string) bool {
	for k, _ := range r {
		if k == name {
			return true
		}
	}

	return false
}

func GetConvolutionLayer(param *pb.V1LayerParameter) (Layer, error) {
	return NewConvolutionLayer(param)
}

func GetPoolLayer(param *pb.V1LayerParameter) (Layer, error) {
	return NewPoolingLayer(param)
}

func GetLRNLayer(param *pb.V1LayerParameter) (Layer, error) {
	return NewLRNLayer(param)
}

func GetReLULayer(param *pb.V1LayerParameter) (Layer, error) {
	return NewReLULayer(param)
}

func GetSigmoidLayer(param *pb.V1LayerParameter) (Layer, error) {
	return NewSigmoidLayer(param)
}

func GetSoftmaxLayer(param *pb.V1LayerParameter) (Layer, error) {
	return NewSoftmaxLayer(param)
}

func GetTanHLayer(param *pb.V1LayerParameter) (Layer, error) {
	return NewTanHLayer(param)
}

func GetInnerProductLayer(param *pb.V1LayerParameter) (Layer, error) {
	return NewInnerProductLayer(param)
}

func GetDropoutLayer(param *pb.V1LayerParameter) (Layer, error) {
	return NewDropoutLayer(param)
}
