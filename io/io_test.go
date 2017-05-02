package io

import (
	"testing"

	pb "github.com/cvley/gocaffe/proto"
)

func TestParse(t *testing.T) {
	data := []byte(`name: "CaffeNet"\r\ninput_dim: 10\r\ninput_dim: 1\r\ninput_dim: 3`)

	msg := &pb.NetParameter{}
	if err := Parse(data, msg); err != nil {
		t.Fatal(err)
	}
}
