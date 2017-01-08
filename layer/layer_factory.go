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
		engine = pb.ConvolutionParameter_Engine_CAFFE
	}

	if engine != pb.ConvolutionParameter_Engine_CAFFE {
		return nil, errors.New("convolution parameter engine not implement")
	}

	return NewConvolutionLayer(param)
}
