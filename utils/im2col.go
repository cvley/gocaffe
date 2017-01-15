package utils

import (
	"errors"

	"github.com/gonum/blas/blas64"
)

var (
	floatZero = float64(0)
)

func Im2col(data []float64, channels, height, width, kernelH, kernelW, padH, padW, strideH, strideW, dilationH, dilationW int) (blas64.General, error) {
	if len(data) < channels*width*height {
		return blas64.General{}, errors.New("mismatch data length with channels*width*height")
	}

	outH := (height + padH*2 - (dilationH*(kernelH-1))/strideH) + 1
	outW := (width + padW*2 - (dilationW*(kernelW-1))/strideW) + 1
	outData := blas64.General{
		Cols:   outW,
		Rows:   outH,
		Stride: channels * kernelH * kernelW,
		Data:   make([]float64, outH*outW*channels*kernelH*kernelW),
	}

	idx := 0
	for channel := 0; channel < channels; channel++ {
		for kRow := 0; kRow < kernelH; kRow++ {
			for kCol := 0; kCol < kernelW; kCol++ {
				inRow := -padH + kRow*dilationH
				for outRow := 0; outRow < outH; outRow++ {
					if inRow >= 0 && inRow < height {
						inCol := -padW + kCol*dilationW
						for outCol := 0; outCol < outW; outCol++ {
							if inCol >= 0 && inCol < width {
								outData.Data[idx] = data[inRow*width+inCol+channel*width*height]
							}
							inCol += strideW
							idx++
						}
					} else {
						for outCol := 0; outCol < outW; outCol++ {
							idx++
						}
					}
					inRow += strideH
				}
			}
		}
	}

	return outData, nil
}

func Col2im(data blas64.General, channels, height, width, kernelH, kernelW, padH, padW, strideH, strideW, dilationH, dilationW int) ([]float64, error) {
	outH := (height + padH*2 - (dilationH*(kernelH-1))/strideH) + 1
	outW := (width + padW*2 - (dilationW*(kernelW-1))/strideW) + 1

	if len(data.Data) < outH*outW*channels*kernelH*kernelW {
		return nil, errors.New("invalid input data")
	}

	output := make([]float64, height*width*channels)

	idx := 0
	for channel := 0; channel < channels; channel++ {
		for kRow := 0; kRow < kernelH; kRow++ {
			for kCol := 0; kCol < kernelW; kCol++ {
				inRow := -padH + kRow*dilationH
				for outRow := 0; outRow < outH; outRow++ {
					if inRow >= 0 && inRow < height {
						inCol := -padW + kCol*dilationW
						for outCol := 0; outCol < outW; outCol++ {
							if inCol >= 0 && inCol < width {
								output[inRow*width+inCol+channel*width*height] = data.Data[idx]
							}
							inCol += strideW
							idx++
						}
					} else {
						for outCol := 0; outCol < outW; outCol++ {
							idx++
						}
					}
					inRow += strideH
				}
			}
		}
	}

	return output, nil
}
