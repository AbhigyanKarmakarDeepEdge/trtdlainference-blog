#include <string>
#include <fstream>
#include <iostream>
#include "trtinfer.cuh"
#include <chrono>
#include "timer.hpp"
#include "NvOnnxParser.h"
#include <memory>
#include <string>

auto sizer = [](nvinfer1::Dims dims){size_t product = 1; for(int i=0 ; i<dims.nbDims ; i++)	product*=dims.d[i];	return product;};		//Helper Lambda

template <typename T>
bool TRTInfer<T>::loadTRTEngine(std::string engine_filename, int DLACore)
{
	std::ifstream engineFile(engine_filename, std::ios::binary);
    if (!engineFile)	{std::cout << "\nError opening engine file: ";	return false;}

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
	engineFile.close();
    params.runtime = nvinfer1::createInferRuntime(params.Logger);

	if(DLACore>=0)	params.runtime->setDLACore(DLACore);	//Have to do this coz for some reason getDLAcore returns -1 when using trtexec generated files, seems like an API bug

    params.engine = params.runtime->deserializeCudaEngine(engineData.data(), fsize); 
    params.context = params.engine->createExecutionContext();
	
	for(int i=0 ; i<params.engine->getNbBindings() ; i++)	
		if(params.engine->bindingIsInput(i))	Ninput_tensors++ ;
		else									Noutput_tensors++;
	
    engineFile.close();
	return true;
}

template <typename T>
void TRTInfer<T>::allocITensors()
{
	input_tensors.resize(Ninput_tensors);
	for(int i=0 ; i<input_tensors.size() ; i++)	
	{
		input_tensors[i].shape = params.engine->getBindingDimensions(i);
		Ialloc = input_tensors[i].allocTensor();	
	}	
}
//
template <typename T>
void TRTInfer<T>::allocOTensors()
{
	output_tensors.resize(Noutput_tensors);
	for(int i=0 ; i<output_tensors.size() ; i++)	
	{
		output_tensors[i].shape = params.engine->getBindingDimensions(i + Ninput_tensors);
		Oalloc = output_tensors[i].allocTensor();	
	}
}
//
template <typename T>
void TRTInfer<T>::allocIOTensors()
{
	allocITensors();
	allocOTensors();
}
//
template <typename T>
void TRTInfer<T>::shallowcopyITensors(TRTInfer<T> &ITensorCarryingObj)
{
	for (auto &i:ITensorCarryingObj.output_tensors)	input_tensors.push_back(i);
}
//
template <typename T>
void TRTInfer<T>::infer(cudaStream_t &stream)
{
	std::vector<void *> buffers;
	
	for(int i=0 ; i<input_tensors.size() ; i++)		buffers.push_back(input_tensors[i].data);
	for(int i=0 ; i<output_tensors.size() ; i++)	buffers.push_back(output_tensors[i].data);
	
	bool success = params.context->enqueueV2(buffers.data(), stream, nullptr);
	if (!success)
	{
       std::cerr << "Inference Error in " << __FILE__ << " at line " << __LINE__ << '\n';
       exit(1);
	}
}
//
template class TRTInfer<float>;
template class TRTInfer<__half>;
//template class TRTInfer<double>;
//
//template class TRTInfer<int>;
//template class TRTInfer<unsigned int>;
//
//template class TRTInfer<long>;
//template class TRTInfer<unsigned long>;
//
//template class TRTInfer<short>;
//template class TRTInfer<unsigned short>;
//
//template class TRTInfer<char>;
//template class TRTInfer<unsigned char>;

template <typename T>
[[nodiscard]] bool tensor<T>::allocTensor()	
{
	if(shapeCorrectness())	return (!(cudaMalloc(&data, sizer(shape) * sizeof(T))));	
	return false;
}

template <typename T>
[[nodiscard]] bool tensor<T>::freeTensor()	
{
	if(data!=nullptr)	return (!cudaFree(data));	
	return false;
}

template <typename T>
[[nodiscard]] bool tensor<T>::shapeCorrectness()
{
	if (shape.nbDims && sizer(shape)>0)	return true;
	return false;
}
