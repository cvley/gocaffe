package net

import (
	"io/ioutil"
	"os"
	"testing"
)

var (
	deployFile = "../imagenet_deploy.prototxt"
	binFile    = "../imagenet_model"
)

func TestCopyTrainedLayersFromFile(t *testing.T) {
	f := &Net{}
	if err := f.CopyTrainedLayersFromFile(binFile); err != nil {
		t.Fatal(err)
	}
}

func TestNet(t *testing.T) {
	f, err := os.Open(deployFile)
	if err != nil {
		t.Fatal(err)
	}

	b, err := ioutil.ReadAll(f)
	if err != nil {
		t.Fatal(err)
	}

	net, err := New(string(b))
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("%+v\n", net)
	t.Logf("%+v\n", net.Parameters)
}
