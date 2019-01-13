#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
//#define HAVE_STRUCT_TIMESPEC //must use this to prevent timespec struct type redefiniton::https://stackoverflow.com/questions/33557506/timespec-redefinition-error/37072163
//#include <pthread.h> //TODO CONVERT PTHREADS TO WINDOWS THREADS & delete define have struct timespec
#include <thread>//cpu threads
#include <cuda.h>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <stdexcept>
#include <iostream>
#include "device_launch_parameters.h"
#include <builtin_types.h>//float3
#include "helper_math.h"


//used for storing 3D position information
//change vector3 to float3?
//struct vector3
//{
//	float x, y, z;
//
//	struct vector3& operator+=(const vector3& rhs) { x += rhs.x; y += rhs.y; z += rhs.z; return *this; }
//	struct vector3& operator+=(const float& k) { x += k; y += k; z += k; return *this; }
//	struct vector3& operator-=(const vector3& rhs) { x -= rhs.x; y -= rhs.y; z -= rhs.z; return *this; }
//	struct vector3& operator-=(const float& k) { x -= k; y -= k; z -= k; return *this; }
//	struct vector3& operator*=(const int& k) { x *= k; y *= k; z *= k; return *this; }
//	struct vector3& operator/=(const int& k) { x /= k; y /= k; z /= k; return *this; }
//};
//vector3 operator+(vector3 lhs, const vector3& rhs) { return lhs += rhs; }
//vector3 operator+(vector3 lhs, const float k) { return lhs += k; }
//vector3 operator+(const float k, vector3 rhs) { return rhs += k; }
//vector3 operator-(vector3 lhs, const vector3& rhs) { return lhs -= rhs; }
//vector3 operator-(vector3 lhs, const float k) { return lhs -= k; }
//vector3 operator-(const float k, vector3 rhs) { return rhs -= k; }
//vector3 operator*(vector3 lhs, const int k) { return lhs *= k; }
//vector3 operator*(const int k, vector3 rhs) { return rhs *= k; }
//vector3 operator/(vector3 lhs, const int k) { return lhs /= k; }
//vector3 operator/(const int k, vector3 rhs) { return rhs /= k; }



//stores a hair strands data
struct hair
{
	float3 startPoint;
	float3 endPoint;
	float3 *interpolatedPoints;//interpolatedPointSize sized array of float3's //the point closest to startingPoint is at [0]
	int interpolatedPointSize;
	
};

//stores a winds data
struct wind
{
	float strength=1.0f;//determines amount of force applied to hair strands, takes [0,1] values
	char axis;//is the origin axis of the wind, i.e X/Y/Z
	char axisSign='+';//is the sign of the origin axis of the wind, i.e +X or -X
	
};

//stores data on attributes of functions where multithreading takes place
struct threadStruct
{
	//int tid, hair *hairPoints, wind w, int smoothing=5
	hair *hairPoints;
	int tid;
	int smoothing;
	int hairPartitionSize;
	wind w;

};


void SaveOriginalSettings(float headRadius, int hairCount, hair * hairPoints, float hairLength);

void HairPointASetter(int hairCount, float headRadius, hair * hairPoints);

void HairInterpolationSetter(int hairCount, hair * hairPoints, int interpolatedPointSize);

void HairPointBSetter(int hairCount, float headRadius, float hairLength, hair * hairPoints);

void ApplyWind(void * thrStr);

void StartWindZ(hair * hairPoint, char sign, float strength);

void StartWindY(hair * hairPoint, char sign, float strength);

void StartWindX(hair * hairPoint, char sign, float strength);

void CollisionDetection(hair * hairPoint);

int JourneyOfAHairStrand(hair * hairPoint, int index, bool dontPrint);

float hairLengthFinder(hair * hairPoint);

void FreeAllCPU(hair * hairPoints);
