#ifndef FILELOC_H
#define FILELOC_H
#include "sourceLoc.h"
#include <string>
namespace file_locations
{
	std::string FullModel_filepath = std::string(project_source_dir) + "trt/G1G2GPU.trt";
	std::string G1_filepath = std::string(project_source_dir) + "trt/G1DLA.trt";
	std::string G2_filepath = std::string(project_source_dir) + "trt/G2GPU.trt";
}
#endif