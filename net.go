package net

import (
	"io/ioutil"
	"log"
	//	"fmt"
	"os"

	"github.com/cvley/gocaffe/layer"
	pb "github.com/cvley/gocaffe/proto"
	"github.com/golang/protobuf/proto"
)

type Net struct {
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

	log.Printf("construct %s", param.GetName())

	for _, layerParam := range param.GetLayers() {
		if layerParam.GetName() == "conv1" {
			l, err := layer.NewConvolutionLayer(layerParam)
			if err != nil {
				log.Println(err)
			}
			log.Printf("%+v", l)
		}
	}

	return nil, nil
}
