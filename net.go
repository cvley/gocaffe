package net

import (
	"io/ioutil"
	"log"
	"os"

	"github.com/cvley/gocaffe/layer"
	pb "github.com/cvley/gocaffe/proto"
	"github.com/golang/protobuf/proto"
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

	for _, layerParam := range param.GetLayers() {
		switch layerParam.GetType() {
		case pb.V1LayerParameter_CONVOLUTION:
			log.Printf("get layer %s", layerParam.GetName())
			l, err := layer.NewConvolutionLayer(layerParam)
			if err != nil {
				log.Println(err)
			}
			log.Printf("%+v", l)
		case pb.V1LayerParameter_POOLING:
		}
	}

	return nil, nil
}
