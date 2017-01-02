package net

import (
	"github.com/cvley/gocaffe/proto"
	"log"
)

type Net struct {
}

func (net *Net) CopyTrainedLayersFromParam(netParam *proto.NetParameter) (*Net, error) {

}

func (net *Net) CopyTrainedLayersFromFile(file string) (*Net, error) {
	param := &proto.NetParameter{}
}
