package layer

import (
	"errors"
	"fmt"
	"log"

	pb "github.com/cvley/proto"
)

func GetConvolutionLayer(param *pb.LayerParameter) (Layer, error) {
	convParam := param.GetConvolutionParam()
	if convParam == nil {
		return nil, errors.New("get convolution param fail")
	}

	engine := convParam.GetEngine()
	if engine == pb.Default_ConvolutionParameter_Engine {
		engine = pb.ConvolutionParameter_CAFFE
	}

	if engine != pb.ConvolutionParameter_CAFFE {
		return nil, errors.New("convolution parameter engine not implement")
	}

	return NewConvolutionLayer(param)
}

func GetPoolLayer(param *pb.LayerParameter) (Layer, error) {
	poolParam := param.GetPoolingParam()
	if poolParam == nil {
		return nil, errors.New("get pooling param fail")
	}

	engine := poolParam.GetEngine()
	if engine == pb.Default_PoolingParameter_Engine {
		engine = pb.PoolingParameter_CAFFE
	}

	if engine != pb.PoolingParameter_CAFFE {
		return nil, errors.New("pooling parameter engine not implement")
	}

	return NewPoolingLayer(param)
}

func GetLRNLayer(param *pb.LayerParameter) (Layer, error) {
	lrnParam := param.GetLrnParam()
	if lrnParam == nil {
		return nil, errors.New("get lrn param fail")
	}

	engine := lrnParam.GetEngine()
	if engine == pb.Default_LRNParameter_Engine {
		engine = pb.LRNParameter_CAFFE
	}

	if engine != pb.LRNParameter_CAFFE {
		return nil, errors.New("lrn parameter engine not implement")
	}

	return NewLRNLayer(param)
}

func GetReLULayer(param *pb.LayerParameter) (Layer, error) {
	reluParam := param.GetReluParam()
	if reluParam == nil {
		return nil, errors.New("get relu param fail")
	}
	engine := reluParam.GetEngine()
	if engine == pb.Default_ReLUParameter_Engine {
		engine = pb.ReLUParameter_CAFFE
	}

	if engine != pb.ReLUParameter_CAFFE {
		return nil, errors.New("relu parameter engine not implement")
	}

	return NewReLULayer(param)
}
