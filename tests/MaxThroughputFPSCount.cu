#include "DLATestCommon.cuh"
#include "GPUTestCommon.cuh"

bool countstart = false;
bool firststart = true;
bool breakDLA = false;
bool dlafirststart[] = {false, false};

unsigned long localEPOCH;
unsigned long gpuFrames=0, dlaFrames[] = {0, 0};

void framecountGPU(std::string filepath_GPUFull, long warmup_time_ms, long infer_time_ms)
{
	if(warmup_time_ms<=0 || infer_time_ms<=0)	{std::cout << "\nInvalid Run duration recieved!!"; return;}
	long counter = 1;
	TRTInfer<> GPU;
	GPU.loadTRTEngine(filepath_GPUFull);
	GPU.allocIOTensors();
	cudaStream_t gpustream;
	cudaStreamCreate (&gpustream);
	HresTimer T("GPU");
	while (++counter)
	{
		if(((std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1))-localEPOCH) >= warmup_time_ms)	{	countstart = true;	}
		if(countstart && firststart) {	T.start_timer();	firststart=false;	counter = 1;	printf("\nTEST STARTED GPU");	dlafirststart[0] = true;	dlafirststart[1] = true;}
		
		GPU.infer(gpustream);
		cudaStreamSynchronize(gpustream);
		
		if(((std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1))-localEPOCH) >= warmup_time_ms + infer_time_ms) {	breakDLA = true;	break;	}
	}
	T.stop_timer(1);
	gpuFrames = counter;
	printf("\nGPU FRAMES : %ld", counter);
	printf("\nGPU FPS = %f\n",  1000*(float)counter/infer_time_ms);
}

void framecountDLA(std::string filepath_G1, std::string filepath_G2, int core, long warmup_time_ms, long infer_time_ms)
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
	while (!breakDLA)
	{
		counter++;
		if(countstart && dlafirststart[core] )	{	counter = 1;	printf("\nTEST STARTED DLA %d ACK", core);	T.start_timer();	dlafirststart[core] = false;}
		
		DLAG1.infer(dlastream);
		cudaStreamSynchronize(dlastream);
		
		G2.shallowcopyITensors(DLAG1);
		G2.infer(gpustream);
		cudaStreamSynchronize(gpustream);
	}
	T.stop_timer(1);
	dlaFrames[core] = counter;
	printf("\nDLA FRAMES Core %d : %ld", core, counter);
	printf("\nDLA FPS = %f\n",  1000*(float)counter/infer_time_ms);
}

void framecountTEST(std::string filepath_GPUFull, std::string filepath_G1, std::string filepath_G2, long warmup_time_ms = 30000 , long infer_time_ms = 100000)
{
	printf("\nTest Warmup = %ld ms\nTest Infer = %ld ms\n" , warmup_time_ms, infer_time_ms);
	::localEPOCH = std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
	
	std::thread GPUFrameTest(framecountGPU, filepath_GPUFull, warmup_time_ms, infer_time_ms);
	
	std::thread HybridTestDLA1(framecountDLA, filepath_G1, filepath_G2, 0, warmup_time_ms, infer_time_ms);
	std::thread HybridTestDLA2(framecountDLA, filepath_G1, filepath_G2, 1, warmup_time_ms, infer_time_ms);
	
	GPUFrameTest.join();
	HybridTestDLA1.join();
	HybridTestDLA2.join();
	
	printf("\nTotal Frames Processed = %lu" , (gpuFrames + dlaFrames[0] + dlaFrames[1]));
	printf("\nThroughput : %f FPS\n", float(1000*(gpuFrames + dlaFrames[0] + dlaFrames[1]))/infer_time_ms);
}

int main()
{		
	//Set warmup to atleast 30000ms or more for ideal behaviour
	
	size_t warmup_duration_ms = 30000, infer_duration_ms = 30000;
	framecountTEST(file_locations::FullModel_filepath, file_locations::G1_filepath, file_locations::G2_filepath, warmup_duration_ms, infer_duration_ms);
}