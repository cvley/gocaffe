## gocaffe

Note: this is just a **Toy** project! Just for fun and learning.

Implement Caffe deep learning framework in pure golang, still in progress.

### Using protocol buffers with Go

To compile the protocol buffer difinition, run `proboc` with the
`--go_out` parameter set to the directory you want to output the
Go code to:

```
protoc --go_out=. *.proto
```

The generated files will be `suffixed.pb.go`. Refer to [golang
protobuf](https://github.com/golang/protobuf) for more information.
