package layer

import (
	"errors"
	"log"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"

	"github.com/cvley/gocaffe/blob"
	pb "github.com/cvley/gocaffe/proto"
)

type convolutionParam struct {
	pad      []int
	kernel   []int
	stride   []int
	dilation []int
}

// ConvLayer implement convolution layer struct.
type ConvLayer struct {
	Bottom []string
	Top    []string

	ConvParam *pb.ConvolutionParameter

	param *convolutionParam

	numOutput int
	group     int
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
	blobprotos := param.GetBlobs()
	var weight, bias *blob.Blob
	var err error
	weight, err = blob.FromProto(blobprotos[0])
	if err != nil {
		return nil, err
	}

	if convParam.GetBiasTerm() {
		bias, err = blob.FromProto(blobprotos[1])
		if err != nil {
			return nil, err
		}
	}

	cParam := &convolutionParam{
		pad:      []int{},
		kernel:   []int{},
		stride:   []int{},
		dilation: []int{},
	}

	for _, v := range convParam.GetPad() {
		cParam.pad = append(cParam.pad, int(v))
	}

	for _, v := range convParam.GetKernelSize() {
		cParam.kernel = append(cParam.kernel, int(v))
	}

	for _, v := range convParam.GetStride() {
		cParam.stride = append(cParam.stride, int(v))
	}

	for _, v := range convParam.GetDilation() {
		cParam.dilation = append(cParam.dilation, int(v))
	}

	convLayer := &ConvLayer{
		Bottom:    param.GetBottom(),
		Top:       param.GetTop(),
		ConvParam: convParam,
		numOutput: int(convParam.GetNumOutput()),
		group:     int(convParam.GetGroup()),
		param:     param,
		axis:      int(convParam.GetAxis()),
		weight:    weight,
		bias:      bias,
	}

	log.Printf("length of weight: %d", weight.Capacity())

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
	dataCols, outW, outH := im2col(bottom, conv.param)

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
		Shape:    []int{bottom.Num(), conv.numOutput, outW, outH},
		Count:    dataCols.Rows * dataCols.Cols,
		Capacity: dataCols.Rows * dataCols.Cols,
	}, nil
}

func im2col(data *blob.Blob, kernelH, kernelW, padH, padW, strideH, strideW, dilationH, dilationW uint32) (blas64.General, int, int) {
	channels := data.Channels()
	width := data.Width()
	height := data.Height()

	outH := (height + int(padH)*2 - int(dilationH*(kernelH-1))/int(strideH)) + 1
	outW := (width + int(padW)*2 - int(dilationW*(kernelW-1))/int(strideW)) + 1
	outData := blas64.General{
		Cols:   int(outH * outW),
		Rows:   channels * int(kernelH*kernelW),
		Stride: channels * int(kernelH*kernelW),
		Data:   make([]float64, outH*outW*channels*int(kernelH*kernelW)),
	}

	idx := 0
	for channel := 0; channel < channels; channel++ {
		for kRow := 0; kRow < int(kernelH); kRow++ {
			for kCol := 0; kCol < int(kernelW); kCol++ {
				inRow := -int(padH) + kRow*int(dilationH)
				for outRow := 0; outRow < outH; outRow++ {
					if inRow >= 0 && inRow < height {
						inCol := -int(padW) + kCol*int(dilationW)
						for outCol := 0; outCol < outW; outCol++ {
							if inCol >= 0 && inCol < width {
								outData.Data[idx] = data.Data[inRow*width+inCol+channel*width*height]
							}
							inCol += int(strideW)
							idx++
						}
					} else {
						for outCol := 0; outCol < int(outW); outCol++ {
							idx++
						}
					}
					inRow += int(strideH)
				}
			}
		}
	}

	return outData, outW, outH
}
