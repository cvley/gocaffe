package layer

import (
	"errors"
	"fmt"
	"log"
	"math"

	"github.com/cvley/gocaffe/blob"
	pb "github.com/cvley/gocaffe/proto"
)

// PoolingLayer struct
type PoolingLayer struct {
	Params *pb.PoolingParameter

	global   bool
	kernelH  int
	kernelW  int
	padH     int
	padW     int
	strideH  int
	strideW  int
	poolType pb.PoolingParameter_PoolMethod
	bottom   []string
	top      []string
	name     string
}

// NewPoolingLayer will construct a pooling layer from parameters
func NewPoolingLayer(params *pb.V1LayerParameter) (Layer, error) {
	log.Println("construct pooling layer")

	name := params.GetName()
	param := params.GetPoolingParam()

	global := param.GetGlobalPooling()

	// check kernel parameters
	kernelSize := param.GetKernelSize()
	kernelH := param.GetKernelH()
	kernelW := param.GetKernelW()
	if global && (kernelSize != 0 || kernelH != 0 || kernelW != 0) {
		return nil, errors.New("With global pooling: true Filter size cannot specified")
	}
	if (kernelSize != 0) == ((kernelH != 0) && (kernelW != 0)) {
		return nil, errors.New("Filter size is kernel_size OR kernel_h and kernel_w; not both")
	}
	if (kernelSize == 0) && (kernelH == 0 && kernelW == 0) {
		return nil, errors.New("For non-square filter both kernel_h and kernel_w are required")
	}

	if kernelSize != 0 {
		kernelH = kernelSize
		kernelW = kernelSize
	}

	// check pad parameter
	pad := param.GetPad()
	padH := param.GetPadH()
	padW := param.GetPadW()
	if pad != 0 && padH != 0 && padW != 0 {
		return nil, errors.New("pad is pad OR pad_h and pad_w are required")
	}
	if pad != 0 {
		padH = pad
		padW = pad
	}

	// check stride parameter
	stride := param.GetStride()
	strideH := param.GetStrideH()
	strideW := param.GetStrideW()
	if stride != 0 && strideH != 0 && strideW != 0 {
		return nil, errors.New("stride is stride OR stride_h and stride_w are required")
	}
	if stride != 0 {
		strideH = stride
		strideW = stride
	}

	// setup pooling parameter
	return &PoolingLayer{
		Params:   param,
		global:   global,
		kernelH:  int(kernelH),
		kernelW:  int(kernelW),
		padH:     int(padH),
		padW:     int(padW),
		strideH:  int(strideH),
		strideW:  int(strideW),
		poolType: param.GetPool(),
		bottom:   params.GetBottom(),
		top:      params.GetTop(),
		name:     name,
	}, nil
}

// Forward does forward pooling process
func (pool *PoolingLayer) Forward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	channels := bottom[0].Channels()
	height := bottom[0].Height()
	width := bottom[0].Width()

	// reset global pooling parameters
	if pool.global {
		pool.kernelH = int(bottom[0].Height())
		pool.kernelW = int(bottom[0].Width())
		pool.padH = 0
		pool.padW = 0
		pool.strideH = 1
		pool.strideW = 1
	}

	pooledHeight := int64(math.Floor(float64(int(height)+2*pool.padH-pool.kernelH)/float64(pool.strideH))) + 1
	pooledWidth := int64(math.Floor(float64(int(width)+2*pool.padW-pool.kernelW)/float64(pool.strideW))) + 1

	// if we have padding, ensure the last pooling starts strictly inside the
	// image (instead of at the padding); otherwise clip the last.
	if pool.padH > 0 || pool.padW > 0 {
		if (pooledHeight-1)*int64(pool.strideH) >= height+int64(pool.padH) {
			pooledHeight--
		}
		if (pooledWidth-1)*int64(pool.strideW) >= width+int64(pool.padW) {
			pooledWidth--
		}
	}

	shape := []int64{bottom[0].Num(), channels, pooledHeight, pooledWidth}
	top, err := blob.New(shape)
	if err != nil {
		return nil, fmt.Errorf("%+v %s", shape, err)
	}

	switch pool.poolType {
	case pb.PoolingParameter_MAX:
		for n := 0; n < int(bottom[0].Num()); n++ {
			for c := 0; c < int(channels); c++ {
				for ph := 0; ph < int(pooledHeight); ph++ {
					for pw := 0; pw < int(pooledWidth); pw++ {
						hStart := ph*pool.strideH - pool.padH
						wStart := pw*pool.strideW - pool.padW
						hEnd := int(height)
						if hStart+pool.kernelH < int(height) {
							hEnd = hStart + pool.kernelH
						}
						wEnd := int(width)
						if wStart+pool.kernelW < int(width) {
							wEnd = wStart + pool.kernelW
						}
						if hStart < 0 {
							hStart = 0
						}
						if wStart < 0 {
							wStart = 0
						}
						poolIndices := []int{n, c, ph, pw}
						for h := hStart; h < hEnd; h++ {
							for w := wStart; w < wEnd; w++ {
								indices := []int{n, c, h, w}
								bData := bottom[0].Get(indices)
								tData := top.Get(poolIndices)
								if bData > tData {
									top.Set(poolIndices, bData)
								}
							}
						}
					}
				}
			}
		}

	case pb.PoolingParameter_AVE:
		for n := 0; n < int(bottom[0].Num()); n++ {
			for c := 0; c < int(channels); c++ {
				for ph := 0; ph < int(pooledHeight); ph++ {
					for pw := 0; pw < int(pooledWidth); pw++ {
						hStart := ph*pool.strideH - pool.padH
						wStart := pw*pool.strideW - pool.padW
						hEnd := int(height)
						if hStart+pool.kernelH < int(height) {
							hEnd = hStart + pool.kernelH
						}
						wEnd := int(width)
						if wStart+pool.kernelW < int(width) {
							wEnd = wStart + pool.kernelW
						}
						if hStart < 0 {
							hStart = 0
						}
						if wStart < 0 {
							wStart = 0
						}
						poolSize := (hEnd - hStart) * (wEnd - wStart)
						poolIndices := []int{n, c, ph, pw}
						for h := hStart; h < hEnd; h++ {
							for w := wStart; w < wEnd; w++ {
								indices := []int{n, c, h, w}
								top.Set(poolIndices, bottom[0].Get(indices))
							}
						}
						tData := top.Get(poolIndices)
						top.Set(poolIndices, tData/float64(poolSize))
					}
				}
			}
		}

	case pb.PoolingParameter_STOCHASTIC:
		return nil, errors.New("stochastic pooling not implemented")
	}

	log.Println(pool.Type(), bottom[0].Shape(), "->", top.Shape())

	return []*blob.Blob{top}, nil
}

// Type of pooling layer
func (pool *PoolingLayer) Type() string {
	return pool.name
}

func (pool *PoolingLayer) Bottom() []string {
	return pool.bottom
}

func (pool *PoolingLayer) Top() []string {
	return pool.top
}
