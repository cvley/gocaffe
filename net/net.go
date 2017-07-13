package net

import (
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/cvley/gocaffe/blob"
	"github.com/cvley/gocaffe/layer"
	"github.com/golang/protobuf/proto"

	pb "github.com/cvley/gocaffe/proto"
)

type Net struct {
	Parameters *pb.NetParameter
	name       string
	inputDim   []int64
	input      []string
	top        string
	layers     []layer.Layer
	layerNames []string
	index      map[string]int
}

func New(text string) (*Net, error) {
	param := &pb.NetParameter{}
	if err := proto.UnmarshalText(text, param); err != nil {
		return nil, err
	}

	// extract input dim
	inputDim := param.GetInputDim()
	inputShape := param.GetInputShape()
	if inputDim == nil && inputShape == nil {
		return nil, errors.New("net prototxt don't have input dim")
	}
	iDim := []int64{}
	if inputDim == nil {
		for _, shape := range inputShape {
			iDim = append(iDim, shape.GetDim()...)
		}
	} else {
		for _, d := range inputDim {
			iDim = append(iDim, int64(d))
		}
	}

	layers := []layer.Layer{}
	names := []string{}
	index := make(map[string]int)
	idx := 0
	if param.GetLayer() != nil {
		//TODO V0 layer support
		return nil, errors.New("V0 layer not support")
	}

	if param.GetLayers() != nil {
		for _, v := range param.GetLayers() {
			l, err := layer.LayerRegister.CreateLayer(v)
			if err != nil {
				log.Println("ERROR create layer", v.GetName(), "fail", err)
				continue
			}
			layers = append(layers, l)
			names = append(names, v.GetName())
			index[v.GetName()] = idx
			idx++
		}
	}

	return &Net{
		Parameters: param,
		name:       param.GetName(),
		inputDim:   iDim,
		input:      param.GetInput(),
		layers:     layers,
		layerNames: names,
		index:      index,
	}, nil
}

func (net *Net) GetInputSize() (int64, int64) {
	//TODO hard code
	return net.inputDim[2], net.inputDim[3]
}

func (net *Net) CopyTrainedLayersFromFile(file string) error {
	f, err := os.Open(file)
	if err != nil {
		return err
	}

	b, err := ioutil.ReadAll(f)
	if err != nil {
		return err
	}

	param := &pb.NetParameter{}
	if err := proto.Unmarshal(b, param); err != nil {
		return err
	}

	return net.CopyTrainedLayersFromParam(param)
}

func (net *Net) CopyTrainedLayersFromParam(param *pb.NetParameter) error {
	for _, layerParam := range param.GetLayers() {
		if layerParam.GetType().String() == "DATA" {
			continue
		}
		l, err := layer.LayerRegister.CreateLayer(layerParam)
		if err != nil {
			return err
		}

		if len(l.Top()) == 0 {
			continue
		}

		log.Printf("%+v\n", l)
		idx, exist := net.index[l.Type()]
		if !exist {
			log.Println("ERROR not found name", l.Type(), "in net parameters")
			continue
		}
		net.layers[idx] = l
	}

	net.Parameters = param
	net.name = param.GetName()
	return nil
}

func (net *Net) Forward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	return net.ForwardFromTo(bottom, len(net.layers))
}

func (net *Net) ForwardFromTo(bottom []*blob.Blob, end int) (top []*blob.Blob, err error) {
	if !net.checkBottomShape(bottom[0]) {
		return nil, fmt.Errorf("bottom shape %v mismatch net input dim %v", bottom[0].Shape(), net.inputDim)
	}

	for i, l := range net.layers {
		log.Println("process from", l.Bottom()[0], "to", l.Top()[0], i)
		top, err = l.Forward(bottom)
		if err != nil {
			return nil, fmt.Errorf("layer forward %s %s %s", l.Type(), l.Bottom()[0], err)
		}
		if i >= end {
			break
		}
		bottom = top
	}

	return top, nil
}

func (net *Net) checkBottomShape(bottom *blob.Blob) bool {
	shape := bottom.Shape()
	if len(shape) != len(net.inputDim) {
		return false
	}

	for i := 1; i < len(shape); i++ {
		if shape[i] != net.inputDim[i] {
			return false
		}
	}

	return true
}

func (net *Net) Name() string {
	return net.name
}
