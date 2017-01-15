package utils

import (
	"errors"
	"fmt"

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
	fmt.Println(outH, outW)

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
								fmt.Println(idx, inRow, width, inCol, kCol, kRow, inRow*width+inCol+channel*width*height)
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

func col2im(data blas64.Vector, channels, height, width, kernelH, kernelW, padH, padW, strideH, strideW, dilationH, dilationW int) blas64.Vector {
	outputH := (height+2*padH-(dilationH*(kernelH-1)+1))/strideH + 1
	outputW := (width+2*padW-(dilationW*(kernelW-1)+1))/strideW + 1
	dataOut := make([]float64, height*width*channels)
	idx := 0
	for channel := 0; channel < channels; channel++ {
		for kernelRow := 0; kernelRow < kernelH; kernelRow++ {
			for kernelCol := 0; kernelCol < kernelW; kernelCol++ {
				inputRow := -padH + kernelRow*dilationH
				for outputRows := outputH; outputRows >= 0; outputRows-- {
					if inputRow >= 0 && inputRow < height {
						inputCol := -padW + kernelCol*dilationW
						for outputCol := outputW; outputCol >= 0; outputCol-- {
							if inputCol >= 0 && inputCol < width {
								index := channel*width*height + inputRow*width + inputCol
								dataOut[index] = data.Data[index]
							}
							idx++
							inputCol += strideW
						}
					}
					inputRow += strideH
				}
			}
		}
	}

	return blas64.Vector{Data: dataOut}
}
