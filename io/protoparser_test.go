package io

import (
	"reflect"
	"testing"
)

func TestItem(t *testing.T) {
	FailCases := []string{
		"testfail",
		"test:fail:",
	}

	for _, c := range FailCases {
		if _, err := NewItem(c); err == nil {
			t.Fatalf("[%s] should fail", c)
		}
	}

	TestCases := map[string]*Item{
		"name: \"Caffe\"": &Item{Key: "name", Val: "Caffe", Type: reflect.String},
		"name:\"Caffe\"":  &Item{Key: "name", Val: "Caffe", Type: reflect.String},
		"input: 10":       &Item{Key: "input", Val: 10, Type: reflect.Int},
		"input:10":        &Item{Key: "input", Val: 10, Type: reflect.Int},
	}
	for k, v := range TestCases {
		item, err := NewItem(k)
		if err != nil {
			t.Fatalf("[%s] fail: %s", k, err)
		}

		if !reflect.DeepEqual(item, v) {
			t.Fatalf("[%s] fail: mismatch item", k)
		}
	}
}
