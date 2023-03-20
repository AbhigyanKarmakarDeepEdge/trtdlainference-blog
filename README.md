Build Project__
cd build && cmake .. && make -j__
It builds 6 executibles__

| Executible            | Description                                                      |
|-----------------------|------------------------------------------------------------------|
| 1xDLAHybridInfer      | 1 thread of DLA hybrid inference                                 |
| 2xDLAHybridInfer      | 2 threads of DLA hybrid inference                                |
| GPUOnlyInfer          | 1 thread of GPU only inference                                   |
| GPUwith1xDLAHybrid    | 1 thread of GPU + DLA hybrid inference                           |
| GPUwith2xDLAHybrid    | 2 threads of GPU + DLA hybrid inference                          |
| MaxThroughputFPSCount | 2 threads of GPU + DLA hybrid inference and 1 thread of GPU only |

