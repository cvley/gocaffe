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

func GetSigmoidLayer(param *pb.LayerParameter) (Layer, error) {
	sigParam := param.GetSigmoidParam()
	if sigParam == nil {
		return nil, errors.New("get sigmoid param fail")
	}
	engine := sigParam.GetEngine()
	if engine == pb.Default_SigmoidParameter_Engine {
		engine = pb.SigmoidParameter_CAFFE
	}

	if engine != pb.SigmoidParameter_CAFFE {
		return nil, errors.New("sigmoid parameter engine not implement")
	}

	return NewSigmoidLayer(param)
}

func GetSoftmaxLayer(param *pb.LayerParameter) (Layer, error) {
	softParam := param.GetSoftmaxParam()
	if softParam == nil {
		return nil, errors.New("get soft param fail")
	}
	engine := softParam.GetEngine()
	if engine == pb.Default_SoftmaxParameter_Engine {
		engine = pb.SoftmaxParameter_CAFFE
	}

	if engine != pb.SoftmaxParameter_CAFFE {
		return nil, errors.New("soft parameter engine not implement")
	}

	return NewSoftmaxLayer(param)
}

func GetTanHLayer(param *pb.LayerParameter) (Layer, error) {
	tanhParam := param.GetTanhParam()
	if tanhParam == nil {
		return nil, errors.New("get tanh param fail")
	}
	engine := tanhParam.GetEngine()
	if engine == pb.Default_TanHParameter_Engine {
		engine = pb.TanHParameter_CAFFE
	}

	if engine != pb.TanHParameter_CAFFE {
		return nil, errors.New("tanh parameter engine not implement")
	}

	return NewTanHLayer(param)
}
