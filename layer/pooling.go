package layer

import (
	"errors"
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
}

// NewPoolingLayer will construct a pooling layer from parameters
func NewPoolingLayer(params *pb.V1LayerParameter) (Layer, error) {
	log.Println("construct pooling layer")

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

	// check pad parameter
	pad := param.GetPad()
	padH := param.GetPadH()
	padW := param.GetPadW()
	if pad != 0 && padH != 0 && padW != 0 {
		return nil, errors.New("pad is pad OR pad_h and pad_w are required")
	}

	// check stride parameter
	stride := param.GetStride()
	strideH := param.GetStrideH()
	strideW := param.GetStrideW()
	if stride != 0 && strideH != 0 && strideW != 0 {
		return nil, errors.New("stride is stride OR stride_h and stride_w are required")
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

	pooledHeight := int(math.Floor(float64(int(height)+2*pool.padH-pool.kernelH)/float64(pool.strideH))) + 1
	pooledWidth := int(math.Floor(float64(int(width)+2*pool.padW-pool.kernelW)/float64(pool.strideW))) + 1

	// if we have padding, ensure the last pooling starts strictly inside the
	// image (instead of at the padding); otherwise clip the last.
	if pool.padH > 0 || pool.padW > 0 {
		if (pooledHeight-1)*pool.strideH >= int(height)+pool.padH {
			pooledHeight--
		}
		if (pooledWidth-1)*pool.strideW >= int(width)+pool.padW {
			pooledWidth--
		}
	}

	top := blob.New()
	shape := []int32{bottom[0].Num(), channels, int32(pooledHeight), int32(pooledWidth)}
	top.Reshape(shape)

	switch pool.poolType {
	case pb.PoolingParameter_MAX:
		for n := 0; n < int(bottom[0].Num()); n++ {
			for c := 0; c < int(channels); c++ {
				for ph := 0; ph < pooledHeight; ph++ {
					for pw := 0; pw < pooledWidth; pw++ {
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
						poolIndices := []int32{int32(n), int32(c), int32(ph), int32(pw)}
						poolIdx := top.Offset(poolIndices)
						for h := hStart; h < hEnd; h++ {
							for w := wStart; w < wEnd; w++ {
								indices := []int32{int32(n), int32(c), int32(h), int32(w)}
								idx := bottom[0].Offset(indices)
								if bottom[0].Data[idx] > top.Data[poolIdx] {
									top.Data[poolIdx] = bottom[0].Data[idx]
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
				for ph := 0; ph < pooledHeight; ph++ {
					for pw := 0; pw < pooledWidth; pw++ {
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
						poolIndices := []int32{int32(n), int32(c), int32(ph), int32(pw)}
						poolIdx := top.Offset(poolIndices)
						for h := hStart; h < hEnd; h++ {
							for w := wStart; w < wEnd; w++ {
								indices := []int32{int32(n), int32(c), int32(h), int32(w)}
								idx := bottom[0].Offset(indices)
								top.Data[poolIdx] += bottom[0].Data[idx]
							}
						}
						top.Data[poolIdx] /= float64(poolSize)
					}
				}
			}
		}

	case pb.PoolingParameter_STOCHASTIC:
		return nil, errors.New("stochastic pooling not implemented")
	}

	return []*blob.Blob{top}, nil
}

// Type of pooling layer
func (pool *PoolingLayer) Type() string {
	return "Pooling"
}
