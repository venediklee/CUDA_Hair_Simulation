//use these define statements for easily manipulating the code
//------------------------ hair related macros
#define HAIRCOUNT 32*3000
#define HEADRADIUS 250.0f
#define HAIRLENGTH 400.0f
#define INTERPOLATION 14 //determines the number of points between the start and end points of the hair
#define SMOOTHING 1 //1 or more, determines the steps the hair will take to reach the maximum point (since the hair doesnt teleport to target position)

//------------------------ wind related macros
#define WIND_AXIS 'Z' //determines axis the wind is coming from
#define WIND_SIGN '+' //determines if the wind is coming from positive or negative direction
#define WIND_STRENGTH 1.0f //[0.0f,1.0f]

//------------------------ render related macros
#define SCREEN_WIDTH 1600
#define SCREEN_HEIGHT 900
#define NUMBER_OF_HAIRS_TO_RENDER (HAIRCOUNT/96)
#define SHOWTOTALERRORCOUNT true
#define PRINT1RANDOMHAIRINFORMATION false //change journey  of a hair strand too

//------------------------ execution related macros
//#############################################         CPU 
#define CPUTHREADCOUNT 8
//#############################################         GPU
#define BLOCKSIZE 32*24
#define BLOCKCOUNT 125//blockCount*blockSize=hairCount -- do not pass boundary values like block size<=1024 (gpu spesific)


#include "HairSimCPU.h"//includes other include statements
#include "kernel.cuh"
#define _CRTDBG_MAP_ALLOC  
#include <crtdbg.h>  
//--------------------- openGL render headers
#include <GL/glew.h>
#include <GLFW/glfw3.h>

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


//render routine
int render(hair* hairpoints,char *title)
{
	GLFWwindow *window;

	// Initialize the library
	if (!glfwInit())
	{
		return -1;
	}

	// Create a windowed mode window and its OpenGL context
	window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, title, NULL, NULL);

	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	// Make the window's context current
	glfwMakeContextCurrent(window);

	glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT); // specifies the part of the window to which OpenGL will draw (in pixels), convert from normalised to pixels
	glMatrixMode(GL_PROJECTION); // projection matrix defines the properties of the camera that views the objects in the world coordinate frame. Here you typically set the zoom factor, aspect ratio and the near and far clipping planes
	glLoadIdentity(); // replace the current matrix with the identity matrix and starts us a fresh because matrix transforms such as glOrpho and glRotate cumulate, basically puts us at (0, 0, 0)
	glOrtho(-SCREEN_WIDTH/2, SCREEN_WIDTH/2, -SCREEN_HEIGHT/2, SCREEN_HEIGHT/2, -100, 100); // essentially set coordinate system
	glMatrixMode(GL_MODELVIEW); // (default matrix mode) modelview matrix defines how your objects are transformed (meaning translation, rotation and scaling) in your world
	glLoadIdentity(); // same as above comment

	//rotate camera since it is not displaying correctly(probably because of xyz axis confusion
	glRotatef(90, 0, 0, 1);
	glRotatef(-180, 1, 0, 0);

	GLfloat lineVertices[NUMBER_OF_HAIRS_TO_RENDER  *3*(2*INTERPOLATION+2)]; /*=
	{
		200, 100, 0,
		100, 300, 0
	};*/

	for (int renderedHair = 0; renderedHair < NUMBER_OF_HAIRS_TO_RENDER ; renderedHair++)
	{
		lineVertices[renderedHair * 3 * (2 * INTERPOLATION + 2)] = hairpoints[renderedHair].startPoint.y;
		lineVertices[renderedHair * 3 * (2 * INTERPOLATION + 2) + 1] = hairpoints[renderedHair].startPoint.z;
		lineVertices[renderedHair * 3 * (2 * INTERPOLATION + 2) + 2] = 0;

		for (int hairInterpolatedPoints = 0; hairInterpolatedPoints < INTERPOLATION; hairInterpolatedPoints++)
		{
			lineVertices[renderedHair * 3 * (2 * INTERPOLATION + 2) + hairInterpolatedPoints * 6 + 3] = hairpoints[renderedHair].interpolatedPoints[hairInterpolatedPoints].y;
			lineVertices[renderedHair * 3 * (2 * INTERPOLATION + 2) + hairInterpolatedPoints * 6 + 4] = hairpoints[renderedHair].interpolatedPoints[hairInterpolatedPoints].z;
			lineVertices[renderedHair * 3 * (2 * INTERPOLATION + 2) + hairInterpolatedPoints * 6 + 5] = 0;

			lineVertices[renderedHair * 3 * (2 * INTERPOLATION + 2) + hairInterpolatedPoints * 6 + 6] = hairpoints[renderedHair].interpolatedPoints[hairInterpolatedPoints].y;
			lineVertices[renderedHair * 3 * (2 * INTERPOLATION + 2) + hairInterpolatedPoints * 6 + 7] = hairpoints[renderedHair].interpolatedPoints[hairInterpolatedPoints].z;
			lineVertices[renderedHair * 3 * (2 * INTERPOLATION + 2) + hairInterpolatedPoints * 6 + 8] = 0;

		}

		lineVertices[renderedHair * 3 * (2 * INTERPOLATION + 2) + 2 * INTERPOLATION * 3 + 3] = hairpoints[renderedHair].endPoint.y;
		lineVertices[renderedHair * 3 * (2 * INTERPOLATION + 2) + 2 * INTERPOLATION * 3 + 4] = hairpoints[renderedHair].endPoint.z;
		lineVertices[renderedHair * 3 * (2 * INTERPOLATION + 2) + 2 * INTERPOLATION * 3 + 5] = 0;
	}
	

	// Loop until the user closes the window
	while(!glfwWindowShouldClose(window))
	{
		glClear(GL_COLOR_BUFFER_BIT);
		glColor3f(1.0, 1.0, 1.0);
		// Render OpenGL here
		glEnable(GL_LINE_SMOOTH);
		//glEnable(GL_LINE_STIPPLE);
		glPushAttrib(GL_LINE_BIT);
		//glLineWidth(1);
		//glLineStipple(1, 0x00FF);
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3, GL_FLOAT, 0, lineVertices);
		glDrawArrays(GL_LINES, 0, NUMBER_OF_HAIRS_TO_RENDER*(INTERPOLATION+2));
		glDisableClientState(GL_VERTEX_ARRAY);
		glPopAttrib();
		//glDisable(GL_LINE_STIPPLE);
		glDisable(GL_LINE_SMOOTH);

		//draw head as a circle
		glColor3f(1.0f, 0.0f, 0.0f);
		glBegin(GL_LINE_LOOP);
		for (int ii = 0; ii < 1000; ii++)
		{
			float theta = 2.0f * 3.1415926f * float(ii) / float(1000);//get the current angle 
			float x = HEADRADIUS * cosf(theta);//calculate the x component 
			float y = HEADRADIUS * sinf(theta);//calculate the y component 
			glVertex2f(x + 0, y + 0);//output vertex 
		}
		glEnd();
		


		// Swap front and back buffers
		glfwSwapBuffers(window);

		// Poll for and process events
		glfwPollEvents();
	}

	glfwTerminate();

	return 1;
}



// main routine that executes
int main()
{
	//TODO simplify the code by using helper_math.h functions lerp, length, normalize


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
	w.axis = WIND_AXIS;
	w.axisSign = WIND_SIGN;
	w.strength = WIND_STRENGTH;

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

	//do the rendering before hair is simulated on CPU 
	render(hairPoints,"hairs before wind is applied (CPU)");


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
	if(SHOWTOTALERRORCOUNT) for (int i = 0; i < hairCount; i++) totalErrorCount += JourneyOfAHairStrand(hairPoints + i, i, true);
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

	//do the rendering after hair is simulated on CPU 
	render(hairPoints, "hairs after wind is applied (CPU)");
	
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
	HairPointASetterGPU(hairCount, headRadius, hairPoints);
	HairInterpolationSetter(hairCount, hairPoints, interpolation);
	HairPointBSetter(hairCount, headRadius, hairLength, hairPoints);
	//save original settings for GPU
	SaveOriginalSettingsGPU(headRadius, hairCount, hairLength);
	
	if (SHOWTOTALERRORCOUNT) for (int i = 0; i < hairCount; i++) totalErrorCount += JourneyOfAHairStrand(hairPoints + i, i, false);
	if (SHOWTOTALERRORCOUNT) std::cout << "##############################################" << std::endl << "##############################################" <<
		std::endl << "TOTAL ERROR COUNT BEFORE WIND(these errors are done on CPU)->" << totalErrorCount << std::endl << std::endl;

	//do the rendering before hair is simulated on GPU 
	render(hairPoints, "hairs before wind is applied (GPU)");

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

	//do the rendering after hair is simulated on CPU 
	render(hairPoints, "hairs after wind is applied (GPU)");

	//cleanup
	for (int i = 0; i < hairCount; i++) gpuErrchk( cudaFree(h_data[i].interpolatedPoints));
	free(h_data);
	FreeAllCPU(hairPoints);
	FreeAllGPU(hairPoints_device);
	return 0;
}
