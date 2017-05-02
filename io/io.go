package io

import (
	"fmt"
	"log"
	"reflect"
	"regexp"
	"strconv"

	pb "github.com/cvley/gocaffe/proto"
)

var (
	reName *regexp.Regexp
)

const (
	protoVarInt  string = "varint"
	protoBytes   string = "bytes"
	protoFixed32 string = "fixed32"

	protoOptType string = "opt"
	protoRepType string = "rep"
)

func init() {
	reName = regexp.MustCompile(`(\w+),\d+,(\w+),name=(\w+)`)
}

func Parse(data []byte, msg *pb.NetParameter) error {
	st := reflect.TypeOf(*msg)
	rv := reflect.ValueOf(msg)
	for i := 0; i < st.NumField(); i++ {
		field := st.Field(i)
		tag := field.Tag.Get("protobuf")
		log.Println(field.Name, field.Type.String(), "Get tag", tag)
		if !reName.Match([]byte(tag)) {
			continue
		}

		value := rv.Elem().Field(i)
		result := reName.FindStringSubmatch(tag)
		typ := result[1]
		//		opt := result[2]
		name := result[3]
		switch typ {
		case protoBytes:
			continue

		case protoVarInt:
			if err := parseVarInt(data, name, value); err != nil {
				log.Println("ERROR parse var int", err)
			}
		default:
			log.Println("ERROR invalid type of tag protobuf", typ)
		}
	}

	log.Printf("xxxxx %+v\n", *msg)

	return nil
}

func parseVarInt(data []byte, name string, value reflect.Value) error {
	if !value.CanSet() {
		return fmt.Errorf("input reflect value cannot set")
	}

	reStr := name + ": (\\d+)"
	re := regexp.MustCompile(reStr)
	if !re.Match(data) {
		return fmt.Errorf("%s not in %s", name, string(data))
	}
	ret := re.FindAllStringSubmatch(string(data), -1)

	// no submatch result
	if len(ret) == 0 {
		return nil
	}

	log.Println("submatch length", len(ret))
	log.Println("reflect range", value.Cap())
	nval := reflect.MakeSlice(value.Type(), len(ret), len(ret))
	value.Set(nval)

	for i, v := range ret {
		val, err := strconv.ParseInt(v[1], 10, 64)
		if err != nil {
			return err
		}
		value.Index(i).SetInt(val)
	}
	return nil
}
