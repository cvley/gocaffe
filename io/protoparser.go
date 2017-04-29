package io

import (
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"strings"
)

var (
	reInt *regexp.Regexp
)

type Item struct {
	Key  string
	Val  interface{}
	Type reflect.Kind
}

func init() {
	reInt = regexp.MustCompile(`(\w+):\s*(\d+)`)
}

func NewItem(input string) (*Item, error) {
	if !strings.Contains(input, ":") {
		return nil, fmt.Errorf("%s is invalid, no colon", input)
	}

	matchs := reInt.FindStringSubmatch(input)
	if len(matchs) == 3 {
		value, err := strconv.ParseInt(matchs[2], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("%s is invalid, convert to int fail %s", input, err)
		}
		return &Item{
			Key:  trim(matchs[1]),
			Val:  int(value),
			Type: reflect.Int,
		}, nil
	}

	values := strings.Split(input, ":")
	if len(values) != 2 {
		return nil, fmt.Errorf("%s is invalid", input)
	}

	return &Item{
		Key:  trim(values[0]),
		Val:  trim(values[1]),
		Type: reflect.String,
	}, nil
}

func trim(input string) string {
	return strings.Trim(input, " \"")
}
