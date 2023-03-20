#ifndef DLACOMMON_H
#define DLACOMMON_H
#include <string>
#include <iostream>
#include "trtinfer.cuh"
#include <thread>
#include "timer.hpp"

#include "filelocs.h"
bool warmup = true;
bool dlafirststart_DLA[] = {false, false};
unsigned long dlaFrames_DLA[] = {0, 0};

unsigned long localEPOCH_DLA;

long warmup_time_ms = 30000, infer_time_ms = 30000;

void G1G2FullHybridInfer(int core, std::string filepath_G1, std::string filepath_G2 )
{
	long counter = 1;
	
	TRTInfer<> DLAG1;
	DLAG1.loadTRTEngine(filepath_G1, core);
	DLAG1.allocIOTensors();
	
	TRTInfer<> G2;
	G2.loadTRTEngine(filepath_G2);
	G2.allocOTensors();
		
	cudaStream_t dlastream, gpustream;
	cudaStreamCreate (&dlastream);
	cudaStreamCreate (&gpustream);
	
	HresTimer T("Hybrid Infer");
	localEPOCH_DLA = std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
	while (++counter)
	{
		if((((std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1)) - localEPOCH_DLA) >= warmup_time_ms) &&warmup)	{	warmup = false; dlafirststart_DLA[core] = true;	}
		if(dlafirststart_DLA[core] && !warmup)	{	counter = 1;	printf("\nTEST STARTED DLA %d ACK", core);	T.start_timer();	dlafirststart_DLA[core] = false;}
		
		DLAG1.infer(dlastream);
		cudaStreamSynchronize(dlastream);
		
		G2.shallowcopyITensors(DLAG1);
		G2.infer(gpustream);
		cudaStreamSynchronize(gpustream);
		
		if(((std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1))-localEPOCH_DLA) >= warmup_time_ms + infer_time_ms) {	break;	}
	}
	T.stop_timer(1);
	printf("\nDLA FRAMES Core %d : %ld", core, counter);
	printf("\nDLA FPS = %f\n",  1000*(float)counter/infer_time_ms);
}

#endif

