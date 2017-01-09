package layer

import (
	"log"

	pb "github.com/cvley/gocaffe"
)

type ConvLayer struct {
}

func NewConvolutionLayer(param *pb.LayerParameter) (*ConvLayer, error) {
}

func (conv *ConvLayer) SetUp(bottom, top []*Blob) {
}

func (conv *ConvLayer) Reshape(bottom, top []*Blob) {
}

func (conv *ConvLayer) Forward(bottom, top []*Blob) {
}

func (conv *ConvLayer) Backward(bottom, top []*Blob, propagateDown []bool) {
	// not implement yet, only forward is enough
}

func (conv *ConvLayer) Type() string {
	return "ConvolutionLayer"
}
