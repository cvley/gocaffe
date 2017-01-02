package net

import (
	"github.com/cvley/gocaffe/caffeproto"
	"github.com/golang/protobuf/proto"
	"io/ioutil"
	"log"
	"os"
)

type Net struct {
}

func (net *Net) CopyTrainedLayersFromParam(netParam *caffeproto.NetParameter) (*Net, error) {

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

	param := &caffeproto.NetParameter{}
	if err := proto.Unmarshal(b, param); err != nil {
		return nil, err
	}

	fmt.Printf("%+v\n", param)

	return nil, nil
}
