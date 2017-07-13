package io

import (
	"testing"
)

func TestReadImageFile(t *testing.T) {
	b, err := ReadImageFile("./111.jpg", 256, 256)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(b.Capacity())
}
