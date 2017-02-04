package layer

import (
	"errors"
	"log"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"

	"github.com/cvley/gocaffe/blob"
	pb "github.com/cvley/gocaffe/proto"
)

// ConvLayer implement convolution layer struct.
type ConvLayer struct {
	Bottom []string
	Top    []string

	ConvParam *pb.ConvolutionParameter

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

// NewConvolutionLayer implements the convolution layer construction from
// parameters.
func NewConvolutionLayer(param *pb.V1LayerParameter) (*ConvLayer, error) {
	convParam := param.GetConvolutionParam()
	if convParam == nil {
		return nil, errors.New("no convolution parameters")
	}

	//TODO: compatibility
	convLayer := &ConvLayer{
		Bottom:    param.GetBottom(),
		Top:       param.GetTop(),
		ConvParam: convParam,
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
		data, err := conv.forward(v)
		if err != nil {
			return nil, err
		}
		top = append(top, data)
	}

	return top, nil
}

// Backward implement the gradient update
func (conv *ConvLayer) Backward(bottom, top []*blob.Blob, propagateDown []bool) {
	// not implement yet, only forward is enough
}

// Type of Layer
func (conv *ConvLayer) Type() string {
	return "ConvolutionLayer"
}

func (conv *ConvLayer) forward(bottom *blob.Blob) (*blob.Blob, error) {
	// TODO: too many params, maybe use conv param is better?
	dataCols, outW, outH := im2col(bottom, conv.kernel[0], conv.kernel[0], conv.pad[0], conv.pad[0], conv.stride[0], conv.stride[0], conv.dilation[0], conv.dilation[0])

	cols := int(conv.kernel[0]*conv.kernel[0]) * int(bottom.Channels())

	// convolution
	w := blas64.General{
		Cols:   cols,
		Rows:   conv.numOutput,
		Stride: cols,
		Data:   conv.weight.Data,
	}
	c := blas64.General{
		Cols:   conv.numOutput,
		Rows:   cols,
		Stride: conv.numOutput,
		Data:   make([]float64, conv.numOutput*cols),
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1.0, dataCols, w, 0.0, c)

	// bias
	b := blas64.General{
		Cols:   1,
		Rows:   conv.numOutput,
		Stride: 1,
		Data:   conv.bias.Data,
	}
	ones := make([]float64, int(outH*outW))
	for i, _ := range ones {
		ones[i] = 1.0
	}
	bMultiplier := blas64.General{
		Cols:   int(outH * outW),
		Rows:   1,
		Stride: int(outH * outW),
		Data:   ones,
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1.0, bMultiplier, b, 0.0, c)

	return &blob.Blob{
		Data:     c.Data,
		Shape:    []int32{bottom.Num(), int32(conv.numOutput), outW, outH},
		Count:    dataCols.Rows * dataCols.Cols,
		Capacity: dataCols.Rows * dataCols.Cols,
	}, nil
}

func im2col(data *blob.Blob, kernelH, kernelW, padH, padW, strideH, strideW, dilationH, dilationW uint32) (blas64.General, int32, int32) {
	channels := int(data.Channels())
	width := uint32(data.Width())
	height := uint32(data.Height())

	outH := (height + padH*2 - (dilationH*(kernelH-1))/strideH) + 1
	outW := (width + padW*2 - (dilationW*(kernelW-1))/strideW) + 1
	outData := blas64.General{
		Cols:   int(outH * outW),
		Rows:   channels * int(kernelH*kernelW),
		Stride: channels * int(kernelH*kernelW),
		Data:   make([]float64, outH*outW*uint32(channels)*kernelH*kernelW),
	}

	idx := 0
	for channel := 0; channel < channels; channel++ {
		for kRow := 0; kRow < int(kernelH); kRow++ {
			for kCol := 0; kCol < int(kernelW); kCol++ {
				inRow := -padH + uint32(kRow)*dilationH
				for outRow := 0; outRow < int(outH); outRow++ {
					if inRow >= 0 && inRow < height {
						inCol := -padW + uint32(kCol)*dilationW
						for outCol := 0; outCol < int(outW); outCol++ {
							if inCol >= 0 && inCol < width {
								outData.Data[idx] = data.Data[inRow*width+inCol+uint32(channel)*width*height]
							}
							inCol += strideW
							idx++
						}
					} else {
						for outCol := 0; outCol < int(outW); outCol++ {
							idx++
						}
					}
					inRow += strideH
				}
			}
		}
	}

	return outData, int32(outW), int32(outH)
}
