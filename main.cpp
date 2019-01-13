//use these define statements for easily manipulating the code
#define HAIRCOUNT 32*1000
#define HEADRADIUS 20.0f
#define HAIRLENGTH 10.0f
#define INTERPOLATION 30
#define SMOOTHING 5 //1 or more
#define SHOWTOTALERRORCOUNT true
#define PRINT1RANDOMHAIRINFORMATION false //change journey  of a hair strand too
//#############################################         CPU 
#define CPUTHREADCOUNT 16
//#############################################         GPU
#define BLOCKSIZE 32*25
#define BLOCKCOUNT 40//blockCount*blockSize=hairCount -- do not pass boundary values like block size<=1024 (gpu spesific)


#include "HairSimCPU.h"//includes other include statements
#include "kernel.cuh"
#define _CRTDBG_MAP_ALLOC  
#include <crtdbg.h>  
//--------------------- cpu timer
#include <windows.h>
double PCFreq = 0.0;
__int64 CounterStart = 0;

void StartCounter()
{
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li))
		std::cout << "QueryPerformanceFrequency failed!\n";

	PCFreq = double(li.QuadPart) / 1000.0;

	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}
double GetCounter()
{
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart - CounterStart) / PCFreq;
}


//TODO ################## performance settings are on && rtc(runtime error check) is off
// main routine that executes
int main()
{
	//TODO simplify the code by using helper_math.h functions lerp, length, normalize
	// TODO check for memory leaks

	int hairCount = HAIRCOUNT;//number of individual hair strands, make it multiple of 32 for equal partition
	float headRadius = HEADRADIUS;//radius of the head(sphere)
	hair *hairPoints;//stores hair data
	hairPoints = (hair*)malloc(sizeof(hair)*hairCount);
	float hairLength = HAIRLENGTH;//length of each hair strand
	int interpolation = INTERPOLATION ;//amount of new points with equal intervals on each hair
	const int CPUThreadCount = CPUTHREADCOUNT;
	int smoothing = SMOOTHING;//determines the amount of smoothing made when ApplyWind is called--see ApplyWind for more info
	
	//create wind(s) to apply--can only apply 1 wind right now
	wind w;
	w.axis = 'Z';
	w.axisSign = '+';
	w.strength = 0.7f;

	/*
	############################################################################################################################
	#############################################         CPU        ###########################################################
	############################################################################################################################
	*/
	srand(static_cast <unsigned> (time(0))); //seed random
	int hairIndexOfRandomHair = (int)rand() % hairCount;
	int totalErrorCount = 0;

	if (hairCount%CPUThreadCount != 0)//ensures equal partition
	{
		throw std::invalid_argument("hairPoints/CPUThreadCount is not an integer when calling main");
		exit(-1);
	}
	
	if (hairLength / (interpolation + 1) <= 0.1f)
	{
		throw std::invalid_argument("can't have hairLength / (interpolation + 1) <= 0.1f");
		exit(-1);
	}

	if (headRadius < 1.0f)
	{
		throw std::invalid_argument("can't have head radius < 1.0f");
		exit(-1);
	}
		
	HairPointASetter(hairCount, headRadius, hairPoints);
	HairInterpolationSetter(hairCount, hairPoints, interpolation);
	HairPointBSetter(hairCount, headRadius, hairLength, hairPoints);

	/*	TODO GPU+CPU (performance improvement):
	*	instead of hairCount, create hairCount/32 hairs, and for each hair create 31 additional copies near that hair & control them together
	*	this will make branching within warps much much less
	*/
	/*	TODO additional (memory improvement):
	*	remove interpolated points which dont change direction of hair, i.e which dont collide with head, to save memory
	*		you can re-add them when a wind blows, this works like LOD
	*/
	SaveOriginalSettings(headRadius, hairCount, hairPoints, hairLength);//i am not using original hair points->can be removed
	
	if (PRINT1RANDOMHAIRINFORMATION)
	{
		std::cout << "AFTER SAVING ORIGINAL SETTINGS" << std::endl;
		JourneyOfAHairStrand(hairPoints + hairIndexOfRandomHair, hairIndexOfRandomHair, false);
		std::cout<<std::endl;
	}
	if(SHOWTOTALERRORCOUNT) for (int i = 0; i < hairCount; i++) totalErrorCount += JourneyOfAHairStrand(hairPoints + i, i, false);
	if (SHOWTOTALERRORCOUNT) std::cout << "##############################################" << std::endl << "##############################################" <<
		std::endl << "TOTAL ERROR COUNT BEFORE WIND->" << totalErrorCount << std::endl << std::endl;
	
	/*############## this is where the real multithreading is ##############*/
	//create cpu threads
	std::thread CPUthreads[CPUThreadCount];
	double time = 0;
	StartCounter();
	for (int i = 0; i < CPUThreadCount; i++)//start multithreaded functions
	{
		//create function attribute
		threadStruct thStruct;
		thStruct.hairPoints = hairPoints;
		thStruct.smoothing = smoothing;
		thStruct.tid = i;
		thStruct.w = w;
		thStruct.hairPartitionSize = hairCount / CPUThreadCount;//equal partition is ensured

		//start cpu threads
		CPUthreads[i] = std::thread(ApplyWind,(void*)&thStruct);
	}
	
	//join threads
	for (int i = 0; i < CPUThreadCount; i++) CPUthreads[i].join();
	time = GetCounter();
	std::cout << "CPU time->" << time << std::endl;
	if (SHOWTOTALERRORCOUNT) for (int i = 0; i < hairCount; i++) totalErrorCount += JourneyOfAHairStrand(hairPoints + i, i, false);
	if (SHOWTOTALERRORCOUNT) std::cout << "##############################################" << std::endl << "##############################################" <<
		std::endl << "TOTAL ERROR COUNT AFTER WIND->" << totalErrorCount << std::endl << std::endl;
	if (PRINT1RANDOMHAIRINFORMATION)
	{
		std::cout << "AFTER APPLYING WIND" << std::endl;
		JourneyOfAHairStrand(hairPoints + hairIndexOfRandomHair, hairIndexOfRandomHair, false);
		std::cout << std::endl;
	}
	
	
	/*
	############################################################################################################################
	#############################################         GPU        ###########################################################
	############################################################################################################################
	*/
	std::cout << "####################################################################################################### " << std::endl;
	std::cout << std::endl << "############################################# STARTING GPU ############################################# " << std::endl;
	std::cout << "####################################################################################################### " << std::endl;

	int blockSize = BLOCKSIZE;
	int blockCount = BLOCKCOUNT;

	totalErrorCount = 0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);

	//re-set hair points
	HairPointASetter(hairCount, headRadius, hairPoints);
	HairInterpolationSetter(hairCount, hairPoints, interpolation);
	HairPointBSetter(hairCount, headRadius, hairLength, hairPoints);
	//save original settings for GPU
	SaveOriginalSettingsGPU(headRadius, hairCount, hairLength);
	
	if (SHOWTOTALERRORCOUNT) for (int i = 0; i < hairCount; i++) totalErrorCount += JourneyOfAHairStrand(hairPoints + i, i, false);
	if (SHOWTOTALERRORCOUNT) std::cout << "##############################################" << std::endl << "##############################################" <<
		std::endl << "TOTAL ERROR COUNT BEFORE WIND(these errors are done on CPU)->" << totalErrorCount << std::endl << std::endl;

	//transfer hairPoints to GPU
	hair *h_data = (hair*)malloc(hairCount * sizeof(hair));
	memcpy(h_data, hairPoints, sizeof(hair)*hairCount);
	for (int i = 0; i < hairCount; i++)
	{
		gpuErrchk( cudaMalloc((void**)&(h_data[i].interpolatedPoints), sizeof(float3)*h_data[i].interpolatedPointSize));
		gpuErrchk(cudaMemcpy(h_data[i].interpolatedPoints, hairPoints[i].interpolatedPoints, hairPoints[i].interpolatedPointSize * sizeof(float3),cudaMemcpyHostToDevice));
	}//interpolated points are now on GPU

	hair *hairPoints_device;
	gpuErrchk(cudaMalloc((void**)&hairPoints_device, hairCount * sizeof(hair)));
	gpuErrchk(cudaMemcpy(hairPoints_device, h_data, hairCount * sizeof(hair), cudaMemcpyHostToDevice));

	//call apply wind --handles wind passing etc.
	cudaEventRecord(start, 0);
	
	ApplyWindV0(hairPoints_device, w, blockSize, blockCount, smoothing);
	
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float cudaTotalTime;
	cudaEventElapsedTime(&cudaTotalTime, start, stop);
	cudaEventDestroy(start); cudaEventDestroy(stop);
	std::cout << "Total GPU Time Spent: " << cudaTotalTime << std::endl;
	
	
	//copy back the hairs
	cudaMemcpy(h_data, hairPoints_device, hairCount * sizeof(hair), cudaMemcpyDeviceToHost);
	//memcpy(hairPoints, h_data, hairCount * sizeof(hair));
	for (int i = 0; i < hairCount; i++)
	{
		hairPoints[i].endPoint = h_data[i].endPoint;
		cudaMemcpy( hairPoints[i].interpolatedPoints, h_data[i].interpolatedPoints, hairPoints[i].interpolatedPointSize * sizeof(float3), cudaMemcpyDeviceToHost);
	}//hair points are now on CPU
	
	if (SHOWTOTALERRORCOUNT) for (int i = 0; i < hairCount; i++) totalErrorCount += JourneyOfAHairStrand(hairPoints + i, i, false);
	if (SHOWTOTALERRORCOUNT) std::cout << "##############################################" << std::endl << "##############################################" <<
		std::endl << "TOTAL ERROR COUNT AFTER WIND->" << totalErrorCount << std::endl << std::endl;
	if (PRINT1RANDOMHAIRINFORMATION)
	{
		std::cout << "AFTER APPLYING WIND" << std::endl;
		JourneyOfAHairStrand(hairPoints + hairIndexOfRandomHair, hairIndexOfRandomHair, false);
		std::cout << std::endl;
	}

	//cleanup
	for (int i = 0; i < hairCount; i++) gpuErrchk( cudaFree(h_data[i].interpolatedPoints));
	free(h_data);
	FreeAllCPU(hairPoints);
	FreeAllGPU(hairPoints_device);
	return 0;
}
