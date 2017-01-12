package layer

import (
	"log"
	"errors"
	"math"

	pb "github.com/cvley/gocaffe"
	"github.com/cvley/gocaffe/blob"
	"github.com/cvley/gocaffe/math"
)

type ConvLayer struct {
	ConvParam *pb.ConvolutionParameter
	IsShared  bool
	Phase     *pb.Phase
	blobs     []*blob.Blob

	forceNdim2col   bool
	channelAxis     int
	numSpatialAxis int
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

func (conv *ConvLayer) SetUp(bottom, top []*blob.Blob) error {
	conv.forceNdim2col = conv.ConvParam.GetForceNdIm2Col()
	conv.channelAxis = bottom[0].CanonicalAxisIndex(int(conv.ConvParam.GetAxis()))

	firstSpaticalAxis := conv.channelAxis + 1
	numAxes := bottom[0].AxesNum()
	conv.numSpatialAxis = numAxes - firstSpaticalAxis
	if conv.numSpaticalAxis < 0 {
		return errors.New("conv layer num spatial axis less than 0")
	}

	bottomDimBlobShape := make([]int32, conv.numSpatialAxis+1)
	spatialDimBlobShape := make([]int32, 1)
	if conv.numSpatialAxis > 1 {
		spatialDimBlobShape = make([]int32, conv.numSpatialAxis)
	}

	// setup filter kernel dimensions (kernel_shape)
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
