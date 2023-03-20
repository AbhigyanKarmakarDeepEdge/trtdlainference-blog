#ifndef TRTINFERCOMMON_H
#define TRTINFERCOMMON_H

#include <NvInfer.h>
#include <vector>
#include <string>
#include <cuda_fp16.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#define SUCCESS            0
#define ALLOC_ERROR       -1

class trt_Logger : public nvinfer1::ILogger
{
public:
	void log(Severity severity, const char *msg) noexcept override	{if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR))	std::cout << msg << "\n";}
};

class trt_VerboseLogger : public nvinfer1::ILogger
{
public:
	void log(Severity severity, const char *msg) noexcept override	
	{
		if (
			(severity == Severity::kERROR)
			|| (severity == Severity::kINTERNAL_ERROR)
			|| (severity == Severity::kWARNING ) 
			|| (severity == Severity::kINFO ) 
			//|| (severity == Severity::kVERBOSE)	//Enable this only if you plan on storing the log because it spits out ungodly amounts of data
			)	
			std::cout << msg << "\n";
	}
};


template <typename T = float>
struct tensor
{
	bool shapeCorrectness();
	bool allocTensor();		
	bool freeTensor();
	
	T* data;
	nvinfer1::Dims shape;
};

struct TRTExecParams
{
	trt_Logger Logger;
	trt_VerboseLogger VerboseLogger;
	nvinfer1::IRuntime *runtime;
	nvinfer1::ICudaEngine *engine;
	nvinfer1::IExecutionContext *context;
};

template <typename T = float>
class TRTInfer
{
public:	//Debug
	size_t Ninput_tensors = 0, Noutput_tensors = 0;
	TRTExecParams params;
	std::vector<tensor<T>> input_tensors, output_tensors;
	bool Ialloc = false, Oalloc = false;
	
//public:
	bool loadTRTEngine(std::string = "model.trt", int = -1);
	void loadONNXModel(std::string = "model.onnx", int = -1);
	void allocITensors();
	void allocOTensors();
	void allocIOTensors();
	void infer(cudaStream_t &);
	void shallowcopyITensors(TRTInfer<T> &);
};

#endif