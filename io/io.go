package io

import (
	"io"
	"log"
	"reflect"
	"regexp"

	pb "github.com/cvley/gocaffe/proto"
)

var (
	nameExp *regexp.Regexp
)

func init() {
	nameExp = regexp.MustCompile(`name=(\w+)`)
}

func Parse(reader io.Reader, msg *pb.NetParameter) error {
	tags := make(map[string]interface{})

	st := reflect.TypeOf(*msg)
	for i := 0; i < st.NumField(); i++ {
		field := st.Field(i)
		tag := field.Tag.Get("protobuf")
		result := nameExp.FindAllStringSubmatch(tag, 1)
		log.Println(result)
		log.Println(tag)
		tags[tag] = 0
	}

	return nil
}
