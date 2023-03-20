#include "DLATestCommon.cuh"

void HybridTest(std::string filepath_G1, std::string filepath_G2)
{
	std::thread DLACore0(G1G2FullHybridInfer, 0, filepath_G1, filepath_G2);
	std::thread DLACore1(G1G2FullHybridInfer, 1, filepath_G1, filepath_G2);
	
	DLACore0.join();
	DLACore1.join();
}

int main()
{	
	HybridTest(file_locations::G1_filepath, file_locations::G2_filepath);
}
