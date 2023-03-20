#include "DLATestCommon.cuh"
#include "GPUTestCommon.cuh"

void GPUwithHybridTest(std::string filepath_GPUFull, std::string filepath_G1, std::string filepath_G2)
{
	std::thread GPU(GPUOnly, filepath_GPUFull);
	std::thread DLACore0(G1G2FullHybridInfer, 0, filepath_G1, filepath_G2);
	
	GPU.join();
	DLACore0.join();
}

int main()
{	
	GPUwithHybridTest(file_locations::FullModel_filepath, file_locations::G1_filepath, file_locations::G2_filepath);
}
