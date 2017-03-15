package net

import (
	"io/ioutil"
	"log"
	"os"

	"github.com/cvley/gocaffe/layer"
	"github.com/golang/protobuf/proto"

	pb "github.com/cvley/gocaffe/proto"
)

type Net struct {
	name           string
	layers         []layer.Layer
	layerNames     []string
	layerNameIndex map[string]int
}

func (net *Net) CopyTrainedLayersFromParam(netParam *pb.NetParameter) (*Net, error) {
	log.Println("Ok")
	return nil, nil
}

func (net *Net) CopyTrainedLayersFromFile(file string) (*Net, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}

	b, err := ioutil.ReadAll(f)
	if err != nil {
		return nil, err
	}
	log.Println(len(b))

	param := &pb.NetParameter{}
	if err := proto.Unmarshal(b, param); err != nil {
		return nil, err
	}

	net.name = param.GetName()
	log.Printf("construct %s", param.GetName())

	// TODO use layer registry
	for _, layerParam := range param.GetLayers() {
		log.Printf("get layer %s, type %s", layerParam.GetName(), layerParam.GetType())
		switch layerParam.GetType() {
		case pb.V1LayerParameter_CONVOLUTION:
			l, err := layer.NewConvolutionLayer(layerParam)
			if err != nil {
				log.Println(err)
			}
			log.Printf("%+v", l)

		case pb.V1LayerParameter_POOLING:
			l, err := layer.NewPoolingLayer(layerParam)
			if err != nil {
				log.Println(err)
			}
			log.Printf("%+v", l)

		case pb.V1LayerParameter_LRN:
			l, err := layer.NewLRNLayer(layerParam)
			if err != nil {
				log.Println(err)
			}
			log.Printf("%+v", l)

		case pb.V1LayerParameter_RELU:
			l, err := layer.NewReLULayer(layerParam)
			if err != nil {
				log.Println(err)
			}
			log.Printf("%+v", l)

		case pb.V1LayerParameter_SIGMOID:
			l, err := layer.NewSigmoidLayer(layerParam)
			if err != nil {
				log.Println(err)
			}
			log.Printf("%+v", l)

		case pb.V1LayerParameter_INNER_PRODUCT:
			l, err := layer.NewInnerProductLayer(layerParam)
			if err != nil {
				log.Println(err)
			}
			log.Printf("%+v", l)

		case pb.V1LayerParameter_DROPOUT:
			l, err := layer.NewDropoutLayer(layerParam)
			if err != nil {
				log.Println(err)
			}
			log.Printf("%+v", l)

		case pb.V1LayerParameter_SOFTMAX_LOSS:
			l, err := layer.NewSoftmaxLayer(layerParam)
			log.Println(layerParam)
			if err != nil {
				log.Println(err)
			}
			log.Printf("%+v", l)
		}
	}

	return nil, nil
}
