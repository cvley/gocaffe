package layer

import (
	"fmt"
	"log"
	"strings"
)

var (
	LayerRegisterer LayerRegistry
)

type Layer interface{}

type Creator func(LayerParameter) Layer

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
