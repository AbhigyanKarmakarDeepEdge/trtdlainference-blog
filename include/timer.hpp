#include <string>
#include <iostream>
#include <chrono>

#ifndef TIMER_H
#define TIMER_H

class HresTimer {
	bool timerused = false;
	bool fps;
public:
    
	HresTimer(std::string s, bool _fps = false) { name = s; fps = _fps;}
    void set_name(std::string s) { name = s; }
    void start_timer() {
		timerused = true;
        cpu_start = clock();
        tstart = std::chrono::high_resolution_clock::now();
    }
    void stop_timer(bool verbose = false) {
        cpu_end = clock();
        tend = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tstart);
        if(verbose)
			if(fps)	std::cout << "\nFPS in " << name << " : " << (1.0/(time_span.count()));		
			else std::cout << "\nTime in " << name << " : " << ((time_span.count()));
    }
	double ret_time()
	{
		if (timerused)	return time_span.count()*1000;
		return -1;
	}

private:
    std::string name;
    clock_t cpu_start, cpu_end;
	std::chrono::duration<double> time_span;
    std::chrono::high_resolution_clock::time_point tstart, tend;
};

#endif