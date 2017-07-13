package layer

import (
	"errors"
	"log"

	"github.com/cvley/gocaffe/blob"
	pb "github.com/cvley/gocaffe/proto"
)

type DataLayer struct {
}

func NewDataLayer(param *pb.V1LayerParameter) (*DataLayer, error) {
	dataParam := param.GetDataParam()
	log.Println(dataParam)
	if dataParam == nil {
		return nil, errors.New("set up data layer fail")
	}

	return nil, nil
}

func (data *DataLayer) Forward(bottom []*blob.Blob) ([]*blob.Blob, error) {
	return nil, nil
}

func (data *DataLayer) Type() string {
	return "Data"
}
