package layer

import (
	"testing"
)

func TestLayerRegister(t *testing.T) {
	t.Log(LayerRegister.LayerTypeList())
}
