package net

import (
	caffeproto "github.com/cvley/gocaffe/proto"
	"github.com/golang/protobuf/proto"
	"io/ioutil"
	"log"
//	"fmt"
	"os"
)

type Net struct {
}

func (net *Net) CopyTrainedLayersFromParam(netParam *caffeproto.NetParameter) (*Net, error) {
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

	param := &caffeproto.NetParameter{}
	if err := proto.Unmarshal(b, param); err != nil {
		return nil, err
	}

	for _, layer := range param.GetLayers() {
		log.Println(layer.GetName(), layer.GetType(), layer.GetBottom(), layer.GetTop())
		log.Println(layer)
	}

	return nil, nil
}
