package main

import (
	"flag"
	"io/ioutil"
	"log"
	"os"

	"github.com/cvley/gocaffe/blob"
	"github.com/cvley/gocaffe/io"
	"github.com/cvley/gocaffe/net"
)

func main() {
	deploy := flag.String("deploy", "", "deploy prototxt file")
	model := flag.String("model", "", "trained model")
	image := flag.String("image", "", "input image")
	flag.Parse()

	if *deploy == "" || *model == "" {
		log.Println("invalid deploy or model file")
		flag.PrintDefaults()
		os.Exit(1)
	}

	if *image == "" {
		log.Println("invalid image file")
		flag.PrintDefaults()
		os.Exit(1)
	}

	f, err := os.Open(*deploy)
	if err != nil {
		log.Println(err)
		os.Exit(1)
	}
	defer f.Close()

	b, err := ioutil.ReadAll(f)
	if err != nil {
		log.Println(err)
		os.Exit(1)
	}

	n, err := net.New(string(b))
	if err != nil {
		log.Println(err)
		os.Exit(1)
	}

	if err := n.CopyTrainedLayersFromFile(*model); err != nil {
		log.Println(err)
		os.Exit(1)
	}

	height, width := n.GetInputSize()
	bottom, err := io.ReadImageFile(*image, int(width), int(height))
	if err != nil {
		log.Println(err)
		os.Exit(1)
	}

	log.Println(bottom.Shape())
	tops, err := n.Forward([]*blob.Blob{bottom})
	if err != nil {
		log.Println("ERROR Forward", err)
		os.Exit(1)
	}

	log.Println(tops[0].GetTop(5))
}
