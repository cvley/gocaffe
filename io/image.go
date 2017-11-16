package io

import (
	"image"
	"io/ioutil"
	"log"
	"os"

	_ "image/jpeg"

	"github.com/cvley/gocaffe/blob"
	"github.com/golang/protobuf/proto"
	"github.com/nfnt/resize"

	pb "github.com/cvley/gocaffe/proto"
)

type Image struct {
	data     []float32
	channels int
	width    int
	height   int
}

func ReadImageFile(file string, width int, height int, mean string) (*blob.Blob, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	meanBlob, err := readMean(mean)
	if err != nil {
		return nil, err
	}

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}

	m := resize.Resize(uint(meanBlob.Width()), uint(meanBlob.Height()), img, resize.Lanczos3)

	// TODO hard code
	//shape := []int64{1, 3, meanBlob.Height(), meanBlob.Width()}
	shape := []int64{1, 3, int64(height), int64(width)}
	datum, err := blob.New(shape)
	if err != nil {
		return nil, err
	}

	for x := 0; x < int(meanBlob.Width()); x++ {
		for y := 0; y < int(meanBlob.Height()); y++ {
			r, g, b, _ := m.At(x, y).RGBA()
			datum.Set([]int{0, 0, y, x}, float64(b>>8)-meanBlob.Get([]int{0, 0, y, x}))
			datum.Set([]int{0, 1, y, x}, float64(g>>8)-meanBlob.Get([]int{0, 1, y, x}))
			datum.Set([]int{0, 2, y, x}, float64(r>>8)-meanBlob.Get([]int{0, 2, y, x}))
		}
	}

	return datum, nil
}

func readMean(file string) (*blob.Blob, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}

	buf, err := ioutil.ReadAll(f)
	if err != nil {
		return nil, err
	}

	b := &pb.BlobProto{}
	if err := proto.Unmarshal(buf, b); err != nil {
		return nil, err
	}

	return blob.FromProto(b)
}
