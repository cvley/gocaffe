package layer

import (
	"log"

	pb "github.com/cvley/gocaffe"
	"github.com/cvley/gocaffe/blob"
	"github.com/cvley/gocaffe/math"
)

type ConvLayer struct {
	ConvParam *pb.ConvolutionParameter
	IsShared  bool
	Phase     *pb.Phase
	blobs     []*blob.Blob
}

func NewConvolutionLayer(param *pb.LayerParameter) *ConvLayer {
	phase := param.GetPhase()
	blobprotos := param.GetBlobs()
	blobs := make([]*blob.Blob, len(blobprotos))
	for i, p := range blobprotos {
		blobs[i].FromProto(p, true)
	}
	return &ConvLayer{
		ConvParam: param.GetConvolutionParam(),
		IsShared:  false,
		Phase:     phase,
		blobs:     blobs,
	}
}

func (conv *ConvLayer) SetUp(bottom, top []*blob.Blob) {
}

func (conv *ConvLayer) Reshape(bottom, top []*blob.Blob) {
}

func (conv *ConvLayer) Forward(bottom, top []*blob.Blob) {
}

func (conv *ConvLayer) Backward(bottom, top []*blob.Blob, propagateDown []bool) {
	// not implement yet, only forward is enough
}

func (conv *ConvLayer) Type() string {
	return "ConvolutionLayer"
}

func im2colNd() {

}

func im2col() {
}
