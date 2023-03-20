#include "GPUTestCommon.cuh"

void GPUTest(std::string filepath_GPUFull)
{
	std::thread GPU(GPUOnly, filepath_GPUFull);
	
	GPU.join();
}


int main()
{	
	GPUTest(file_locations::FullModel_filepath);
}
