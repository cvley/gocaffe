package io

import (
	"image"
	"log"
	"os"

	_ "image/jpeg"

	"github.com/cvley/gocaffe/blob"
	"github.com/nfnt/resize"
)

type Image struct {
	data     []float32
	channels int
	width    int
	height   int
}

func ReadImageFile(file string, width int, height int) (*blob.Blob, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}

	m := resize.Resize(uint(width), uint(height), img, resize.Lanczos3)

	// TODO hard code
	shape := []int64{1, 3, int64(height), int64(width)}
	datum, err := blob.New(shape)
	if err != nil {
		return nil, err
	}
	log.Println(height, width)

	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			r, g, b, _ := m.At(x, y).RGBA()
			datum.Set([]int{0, 0, y, x}, float64(r>>8))
			datum.Set([]int{0, 1, y, x}, float64(g>>8))
			datum.Set([]int{0, 2, y, x}, float64(b>>8))
		}
	}

	return datum, nil
}
