package io

import (
	"bytes"
	"testing"

	pb "github.com/cvley/gocaffe/proto"
)

func TestParse(t *testing.T) {
	data := []byte(`name: "CaffeNet"`)

	msg := &pb.NetParameter{}
	if err := Parse(bytes.NewReader(data), msg); err != nil {
		t.Fatal(err)
	}
}
