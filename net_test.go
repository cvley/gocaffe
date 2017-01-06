package net

import (
	"testing"
)

var (
	ProtoFile = "imagenet_deploy.prototxt"
	BinFile = "imagenet_model"
)

func TestCopyTrainedLayersFromFile(t *testing.T) {
	f := &Net{}
	if _, err := f.CopyTrainedLayersFromFile(BinFile); err != nil {
		t.Fatal(err)
	}
}
