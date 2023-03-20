PROGRAM=$1

OUTPUT=$(pwd)/nsight-profile-`date +%Y%m%d%H%M%S`.qdstrm
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

sudo nsys profile --trace=osrt,cuda,nvtx,cublas,cudnn,opengl,nvmedia --cudabacktrace=true --accelerator-trace=nvmedia --backtrace=dwarf -o $OUTPUT $PROGRAM
