package layer

import (
	"fmt"
	"log"
	"strings"

	pb "github.com/cvley/proto"
)

var (
	LayerRegisterer LayerRegistry
)

type Layer interface {
	SetUp(bottom, top []*Blob)
	Reshape(bottom, top []*Blob)
	Forward(bottom, top []*Blob) float64
	Backward(bottom, top []*Blob, propagateDown []bool)
	Type() string
}

type Creator func(*pb.LayerParameter) (Layer, error)

type LayerRegistry map[string]Creator

func init() {
	LayerRegisterer = make(LayerRegistry)
}

func (r *LayerRegistry) AddCreator(tp string, creator Creator) error {
	if r.layerExist(tp) {
		return fmt.Errorf("Layer type %s already registered.")
	}
	r[tp] = creator
	return nil
}

func (r *LayerRegistry) CreateLayer(param *pb.LayerParameter) (Layer, error) {
	log.Printf("Creating layer %s", param.GetName())
	tp := param.GetType()
	if !r.layerExist(tp) {
		return nil, fmt.Errorf("layer %s not exist", tp)
	}

	return r[tp](param)
}

func (r *LayerRegistry) LayerTypeList() []string {
	result := []string{}
	for name, _ := range r {
		result = append(result, name)
	}
	return result
}

func (r *LayerRegistry) LayerTypeListString() string {
	typeList := r.LayerTypeList()
	return strings.Join(typeList, ", ")
}

func (r *LayerRegistry) layerExist(name string) bool {
	for k, _ := range r {
		if k == name {
			return true
		}
	}

	return false
}
