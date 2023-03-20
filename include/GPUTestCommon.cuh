#ifndef GPUCOMMON_H
#define GPUCOMMON_H
#include <string>
#include <iostream>
#include "trtinfer.cuh"
#include <thread>
#include "timer.hpp"

#include "filelocs.h"

bool countstart_GPU = false;
bool firststart_GPU = true;

unsigned long localEPOCH_GPU;

long warmup_time_ms_GPU = 30000, infer_time_ms_GPU = 30000;

void GPUOnly(std::string filepath_GPUFull)
{
	if(warmup_time_ms_GPU<=0 || infer_time_ms_GPU<=0)	{std::cout << "\nInvalid Run duration recieved!!"; return;}
	printf("\nTest Warmup = %ld ms\nTest Infer = %ld ms\n" , warmup_time_ms_GPU, infer_time_ms_GPU);
	long counter = 1;
	TRTInfer<> GPU;
	GPU.loadTRTEngine(filepath_GPUFull);
	GPU.allocIOTensors();
	cudaStream_t gpustream;
	cudaStreamCreate (&gpustream);
	HresTimer T("GPU");
	
	localEPOCH_GPU = std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
	while (++counter)
	{
		if(((std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1))-localEPOCH_GPU) >= warmup_time_ms_GPU)	{	countstart_GPU = true;	}
		if(countstart_GPU && firststart_GPU) {	T.start_timer();	firststart_GPU=false;	counter = 1;	printf("\nTEST STARTED GPU");	}
		
		GPU.infer(gpustream);
		cudaStreamSynchronize(gpustream);
		
		if(((std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1))-localEPOCH_GPU) >= warmup_time_ms_GPU + infer_time_ms_GPU) {	break;	}
	}
	T.stop_timer(1);
	printf("\nGPU FRAMES : %ld", counter);
	printf("\nGPU FPS = %f\n",  1000*(float)counter/infer_time_ms_GPU);
}
#endif