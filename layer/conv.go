package layer

import (
	"errors"
	"log"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"

	"github.com/cvley/gocaffe/blob"
	pb "github.com/cvley/gocaffe/proto"
	"github.com/cvley/gocaffe/utils"
)

type ConvLayer struct {
	ConvParam *pb.ConvolutionParameter
	Bottom    []string
	Top       []string

	numOutput int
	group     int
	pad       []uint32
	kernel    []uint32
	stride    []uint32
	dilation  []uint32
	axis      int
	weight    *blob.Blob
	bias      *blob.Blob
}

func NewConvolutionLayer(param *pb.V1LayerParameter) (*ConvLayer, error) {
	convParam := param.GetConvolutionParam()
	if convParam == nil {
		return nil, errors.New("no convolution parameters")
	}

	//TODO: compatibility
	convLayer := &ConvLayer{
		ConvParam: convParam,
		Bottom:    param.GetBottom(),
		Top:       param.GetTop(),
		numOutput: int(convParam.GetNumOutput()),
		group:     int(convParam.GetGroup()),
		pad:       convParam.GetPad(),
		kernel:    convParam.GetKernelSize(),
		stride:    convParam.GetStride(),
		dilation:  convParam.GetDilation(),
		axis:      int(convParam.GetAxis()),
		weight:    blob.New(),
		bias:      blob.New(),
	}

	blobprotos := param.GetBlobs()
	if err := convLayer.weight.FromProto(blobprotos[0], true); err != nil {
		return nil, err
	}
	if convParam.GetBiasTerm() {
		if err := convLayer.bias.FromProto(blobprotos[1], true); err != nil {
			return nil, err
		}
	}

	log.Printf("length of weight: %d", len(convLayer.weight.Data))

	return convLayer, nil
}

// Forward implement the calculation from bottom to top
func (conv *ConvLayer) Forward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	top := []*blob.Blob{}
	for _, v := range bottom {
		data, err := conv.forwardGemm(v, conv.weight.Data)
		if err != nil {
			return nil, err
		}
		if err := conv.forwardBias(data, conv.bias.Data); err != nil {
			return nil, err
		}
		top = append(top, data)
	}

	return top, nil
}

func (conv *ConvLayer) Backward(bottom, top []*blob.Blob, propagateDown []bool) {
	// not implement yet, only forward is enough
}

func (conv *ConvLayer) Type() string {
	return "ConvolutionLayer"
}

func (conv *ConvLayer) forwardGemm(bottom *blob.Blob, weight []float64) (*blob.Blob, error) {
	channels := int(bottom.Channels())
	height := int(bottom.Height())
	width := int(bottom.Width())
	data, err := utils.Im2col(bottom.Data, channels, height, width,
		int(conv.kernel[0]), int(conv.kernel[0]), int(conv.pad[0]), int(conv.pad[0]),
		int(conv.stride[0]), int(conv.stride[0]), int(conv.dilation[0]), int(conv.dilation[0]))
	if err != nil {
		return nil, err
	}

	w := blas64.General{
		Rows:   conv.numOutput,
		Cols:   int(conv.kernel[0] * conv.kernel[0] * channels),
		Stride: int(conv.kernel[0] * conv.kernel[0] * channels),
		Data:   weight,
	}
	c := blas64.General{
		Rows:   conv.numOutput,
		Cols:   data.Cols,
		Stride: data.Cols,
		Data:   make([]float64, conv.numOutput*data.Cols),
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, float64(1), w, data, float64(0), c)

	return &blob.Blob{
		Data:     c.Data,
		Shape:    []int32{int32(1), int32(1), int32(data.Rows), int32(data.Cols)},
		Count:    data.Rows * data.Cols,
		Capacity: data.Rows * data.Cols,
	}, nil
}

func (conv *ConvLayer) forwardBias(top *blob.Blob, bias []float64) error {
	out := blas64.General{
		Rows:   top.Height(),
		Cols:   top.Width(),
		Stride: top.Height() * top.Width(),
		Data:   top.Data,
	}
	b := blas64.General{
		Rows:   conv.numOutput,
		Cols:   1,
		Stride: 1,
		Data:   bias,
	}
	multiplier := blase.General{
		Rows:   conv.numOutput,
		Cols:   1,
		Stride: 1,
		Data:   bias,
	}
	out, err := utils.GocaffeGemm(blas.NoTrans, blas.NoTrans, float64(1), b, multiplier, float64(1), out)
	if err != nil {
		return err
	}
}
