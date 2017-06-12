package layer

import (
	"errors"
	"log"

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
	if blobprotos != nil {
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
		log.Printf("length of weight: %d", weight.Capacity())

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
		param:     cParam,
		axis:      int(convParam.GetAxis()),
		weight:    weight,
		bias:      bias,
	}

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
	// convolution
	convBlob, err := conv.conv(bottom)
	if err != nil {
		return nil, err
	}

	// bias
	convBlob.Add(conv.bias, blob.ToData)

	return convBlob, nil
}

func (conv *ConvLayer) conv(data *blob.Blob) (*blob.Blob, error) {
	num := data.Num()
	channels := data.Channels()
	width := data.Width()
	height := data.Height()

	outH := conv.param.getOutputH(height)
	outW := conv.param.getOutputW(width)

	shape := []int{data.Num(), conv.numOutput, outH, outW}
	result, err := blob.New(shape)
	if err != nil {
		return nil, err
	}

	// TODO: simple and naive
	for n := 0; n < num; n++ {
		for o := 0; o < conv.numOutput; o++ {
			for c := 0; c < channels; c++ {
				for h := 0; h < outH; h++ {
					for w := 0; w < outW; w++ {
						sH, eH := conv.param.rangeH(h)
						sW, eW := conv.param.rangeW(w)
						var sum float64
						for y := sH; y < eH; y++ {
							for x := sW; x < eW; x++ {
								if y < 0 || x < 0 {
									continue
								}
								sum += data.Get([]int{n, c, y, x}, blob.ToData) * conv.weight.Get([]int{n, o, y, x}, blob.ToData)
							}
						}
						result.Set([]int{n, o, h, w}, sum, blob.ToData)
					}
				}
			}
		}
	}

	return result, nil
}

func (c *convolutionParam) getOutputH(h int) int {
	return (h+c.pad[0]*2-(c.dilation[0]*(c.kernel[0]-1)+1))/c.stride[0] + 1
}

func (c *convolutionParam) getOutputW(w int) int {
	return (w+c.pad[1]*2-(c.dilation[1]*(c.kernel[1]-1)+1))/c.stride[1] + 1
}

func (c *convolutionParam) rangeH(h int) (int, int) {
	r := (c.dilation[0]*(c.kernel[0]-1) + 1) / 2
	return h - r, h + r
}

func (c *convolutionParam) rangeW(w int) (int, int) {
	r := (c.dilation[1]*(c.kernel[1]-1) + 1) / 2
	return w - r, w + r
}
