package layer

import (
	"fmt"
	"strings"
)

type Layer interface{}

type Creator func(LayerParameter) Layer

type LayerRegistry struct {
	registry map[string]Creator
}

func (r *LayerRegistry) AddCreator(tp string, creator Creator) {
	r.registry[tp] = creator
}

func (r *LayerRegistry) CreateLayer(param *pb.LayerParameter) error {
}

func (r *LayerRegistry) LayerTypeList() []string {
}


func (r *LayerRegistry) LayerTypeListString() string {
	typeList := r.LayerTypeList()
	return strings.Join(typeList, ",")
}
