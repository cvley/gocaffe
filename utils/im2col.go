package utils

import (
	"github.com/gonum/blas/blas64"
)

var (
	floatZero = float64(0)
)

func Im2col(data blas64.Vector, channels, height, width, kernelH, kernelW, padH, padW, strideH, strideW, dilationH, dilationW int) blas64.Vector {
	outputH := (height + padH*2 - (dilationH*(kernelH-1))/strideH) + 1
	outputW := (width + padW*2 - (dilationW*(kernelW-1))/strideW) + 1

	dataCol := []float64{}
	for channel := 0; channel < channels; channel++ {
		for kernelRow := 0; kernelRow < kernelH; kernelRow++ {
			for kernelCol := 0; kernelCol < kernelW; kernelCol++ {
				inputRow := -padH + kernelRow*dilationH
				for outputRows := outputH; outputRows >= 0; outputRows-- {
					if inputRow >= 0 && inputRow < height {
						inputCol := -padW + kernelCol*dilationW
						for outputCol := outputW; outputCol >= 0; outputCol-- {
							if inputCol >= 0 && inputCol < Width {
								index := channel*height*width + (inputRow*width + inputCol)
								dataCol = append(dataCol, data.Data[index])
							} else {
								dataCol = append(dataCol, floatZero)
							}
							inputCol += strideW
						}
					} else {
						for outputCols := outputW; outputCols >= 0; outputCols-- {
							dataCol = append(dataCol, floatZero)
						}
					}
					inputRow += strideH
				}
			}
		}
	}

	return blas64.Vector{Data: dataCol}
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
