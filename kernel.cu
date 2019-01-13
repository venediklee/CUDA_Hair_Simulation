
#include "kernel.cuh"
#define PI 3.14159265358979f

//these are original values, DO NOT FORGET TO SAVE THEM
//TODO check if constant memory is the best choice for these
__constant__ int oHairCount_device;
__constant__ float oHeadRadius_device;
__constant__ float oHairLength_device;

int oHairCount_kernel;
float oHeadRadius_kernel;
float oHairLength_kernel;





//used for setting up the scene
//head position is assumed to be at 0
//headRadius is radius of the head, which is spherical
//hairLength is length of each hairstrand
void SaveOriginalSettingsGPU(float headRadius, int hairCount, float hairLength)
{
	//set the values
	gpuErrchk(cudaMemcpyToSymbol(oHeadRadius_device, &headRadius, sizeof(float)));
	gpuErrchk(cudaMemcpyToSymbol(oHairCount_device, &hairCount, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol(oHairLength_device, &hairLength, sizeof(float)));
	
	oHairCount_kernel = hairCount;
	oHeadRadius_kernel = headRadius;
	oHairLength_kernel = hairLength;
}

//hair points is pointer to hair data
//wind origin is a char {+-X, +-Y, +-Z}, denotes the origin point of the wind, i.e wind blows from wind origin towards head
//wind strength is a float used for determining the amount each individual hair moves-rotates
//hairPartitionSize is the hairs that each 
//smoothing is the amount of steps required for each hair strand to reach peak position
//recieves hairCount, headRadius from global variable(s) oHairCount, oHeadRadius
//TODO additional feature: can use this routine multiple times without stopping all winds
//returns void*(must), changes the direction-rotation of hair OVER 'TIME' UP TO PEAK POSITION depending on smoothing by calling startWind method on GPU
void ApplyWindV0(hair *hairPoints,wind w, int blockSize, int blockCount, int smoothing)
{
	if (blockSize*blockCount != oHairCount_kernel)
	{
		std::cout<<"block size * block count != hair count when calling ApplyWindV0"<<std::endl;
		//exit(-1);
		return;
	}
	
	if (w.strength == 0)
	{
		return;
	}
	if (hairPoints == NULL)
	{
		std::cout << "hairPoints is not malloc'd when calling ApplyWindV0";
		//exit(-1);
		return;
	}
	if ((w.axis != 'X' && w.axis != 'Y' && w.axis != 'Z') || (w.axisSign != '+' && w.axisSign != '-'))
	{
		std::cout << "wind is not set correctly when calling ApplyWindV0";
		//exit(-1);
		return;
	}

	for (int j = 0; j<smoothing; j++)//at each smoothing step
	{
		float smoothedStrength = w.strength*(j + 1) / smoothing;
		//call appropriate function based on wind direction
		if (w.axis == 'X')
		{
			std::cout << "X winds are not implemented yet" << std::endl; return;
		}
		else if (w.axis == 'Y')
		{
			std::cout << "Y winds are not implemented yet" << std::endl; return;
		}
		else//w.axis=='Z'
		{
			StartWindZV0 <<< blockCount, blockSize >>> (hairPoints, w.axisSign, smoothedStrength);
			gpuErrchk(cudaPeekAtLastError());
			CollisionDetectionV0 <<< blockCount,blockSize>>>(hairPoints);
			gpuErrchk(cudaPeekAtLastError());
		}
	}

}


//each thread does work on a single hair
//gets hairPoints, hairPartitionSize, wind axis sign, wind strength
//receives hair length from oHairLength--saved@ SaveOriginalSettings
//hair point is pointer to ONE hair strand data
//wind axis sign determines the clockwise-counterclokwise'ness of hair rotation
//wind strength determines the angle of hair rotation
//returns void, changes positions of ONE hair strand based on the type of wind
//other versions are StartWindY, StartWindX
__global__ void StartWindZV0(hair *hairPoints, char sign, float strength)
{
	//look at applywind()--cpu implementation-- for more info

	int tid = blockDim.x*blockIdx.x + threadIdx.x;

	//change y,z points, x is always same
	//if sign is +, z point gets decreased vice versa
	//find effective wind strength--same on all points of the hair
	float effectiveStrength = (0.75f + (hairPoints[tid].startPoint.z) / (4 * oHeadRadius_device))*strength;



	//TODO performance improvement: instead of calculating Zdist&y point seperataly, do sin cos calculations(like rotation)
	//calculate nEndPoint
	float3 nEndPoint;
	nEndPoint.x = hairPoints[tid].startPoint.x;
	nEndPoint.y = (hairPoints[tid].startPoint.y - hairPoints[tid].endPoint.y)*effectiveStrength + hairPoints[tid].endPoint.y;
	float Zdist = sqrtf(oHairLength_device*oHairLength_device - (nEndPoint.y - hairPoints[tid].startPoint.y)*(nEndPoint.y - hairPoints[tid].startPoint.y));

	if (sign == '+')
	{
		nEndPoint.z = hairPoints[tid].startPoint.z - Zdist;
	}
	else//sign == '-'
	{
		nEndPoint.z = hairPoints[tid].startPoint.z + Zdist;
	}

	float3 * nNonSmoothInterpolatedPoints, *nSmoothInterpolatedPoints;
	nNonSmoothInterpolatedPoints= (float3*)malloc(sizeof(float3)*hairPoints[tid].interpolatedPointSize);
	nSmoothInterpolatedPoints = (float3*)malloc(sizeof(float3)*hairPoints[tid].interpolatedPointSize);
	//calculate nNonSmoothInterpolatedPoints for each interpolated point, then set nSmoothInterpolatedPoints
	for (int i = 0; i < hairPoints[tid].interpolatedPointSize; i++)
	{
		//smoothPoint calculations
		nNonSmoothInterpolatedPoints[i] = (nEndPoint - hairPoints[tid].startPoint)*((float)(i + 1)) / ((float)(hairPoints[tid].interpolatedPointSize + 1)) + hairPoints[tid].startPoint;
		float nRad = atan2f(nNonSmoothInterpolatedPoints[i].y - hairPoints[tid].startPoint.y, nNonSmoothInterpolatedPoints[i].z - hairPoints[tid].startPoint.z);//the angle(in radians) of nonSmoothInterpolatedPoint[i]-startingPoint from the +Z axis(counterclockwise)
		float rad = atan2f(hairPoints[tid].interpolatedPoints[i].y - hairPoints[tid].startPoint.y, hairPoints[tid].interpolatedPoints[i].z - hairPoints[tid].startPoint.z);//the angle(in radians) of interpolatedPoints[i]-startingPoint from +Z axis(counterclockwise)
		float YZdistToStart = sqrtf((nNonSmoothInterpolatedPoints[i].y - hairPoints[tid].startPoint.y)*(nNonSmoothInterpolatedPoints[i].y - hairPoints[tid].startPoint.y)
			+ (nNonSmoothInterpolatedPoints[i].z - hairPoints[tid].startPoint.z)*(nNonSmoothInterpolatedPoints[i].z - hairPoints[tid].startPoint.z));

		nSmoothInterpolatedPoints[i].x = nNonSmoothInterpolatedPoints[i].x;
		nSmoothInterpolatedPoints[i].y = YZdistToStart * sinf((nRad - rad)*(i + 1) / (hairPoints[tid].interpolatedPointSize + 1) + rad) + hairPoints[tid].startPoint.y;//equally divides the angle between nonSmoothInterpolatedPoints & interpolatedPoints, then sets the angle of smoothPoint[i] as i'th step between nonSmoothInterpolatedPoint& interpolatedPoints 
		nSmoothInterpolatedPoints[i].z = YZdistToStart * cosf((nRad - rad)*(i + 1) / (hairPoints[tid].interpolatedPointSize + 1) + rad) + hairPoints[tid].startPoint.z;
	}


	//move hair points
	float3 nVector;//used for finding vector from one point to another
	if (hairPoints[tid].interpolatedPointSize>0) hairPoints[tid].interpolatedPoints[0] = nSmoothInterpolatedPoints[0];
	for (int i = 1; i < hairPoints[tid].interpolatedPointSize; i++)
	{
		//find NORMALIZED vector from interpoaltedPoints[i-1] to nSmoothInterpolatedPoints[i]
		nVector = normalize(nSmoothInterpolatedPoints[i] - hairPoints[tid].interpolatedPoints[i - 1]);

		//set interpolatedPoints[i] to interpolatedPoints[i-1] + nVector * hairLength/(intepolatedPointSize+1)
		hairPoints[tid].interpolatedPoints[i] = hairPoints[tid].interpolatedPoints[i - 1] + (nVector / (float)(hairPoints[tid].interpolatedPointSize + 1))*sqrtf(
			(nEndPoint.y - hairPoints[tid].startPoint.y)*(nEndPoint.y - hairPoints[tid].startPoint.y) + (nEndPoint.z - hairPoints[tid].startPoint.z)*(nEndPoint.z - hairPoints[tid].startPoint.z));
	}

	//set endPoint
	if (hairPoints[tid].interpolatedPointSize > 0)
	{
		nVector = normalize(nEndPoint - hairPoints[tid].interpolatedPoints[hairPoints[tid].interpolatedPointSize - 1]);
		hairPoints[tid].endPoint = hairPoints[tid].interpolatedPoints[hairPoints[tid].interpolatedPointSize - 1] + nVector / (float)(hairPoints[tid].interpolatedPointSize + 1)*sqrtf(
			(nEndPoint.y - hairPoints[tid].startPoint.y)*(nEndPoint.y - hairPoints[tid].startPoint.y) + (nEndPoint.z - hairPoints[tid].startPoint.z)*(nEndPoint.z - hairPoints[tid].startPoint.z));
	}
	else
	{
		//no interpolation:=the hair will be a single line
		hairPoints[tid].endPoint = nEndPoint;
	}
	
	//correct points
	float angleInRad = atan2f((hairPoints[tid].endPoint - hairPoints[tid].startPoint).y, (hairPoints[tid].endPoint - hairPoints[tid].startPoint).z);//angle between current endPoint& +Z axis counterclockwise
	float nAngleInRad = atan2f((nEndPoint - hairPoints[tid].startPoint).y, (nEndPoint - hairPoints[tid].startPoint).z);	//angle between nEndPoint& +Z axis counterclockwise

	float offsetAngleInRad = nAngleInRad - angleInRad;//rotate the hair this much counterclockwise
	
	/*	
	2D rotating of a point around origin counterclockwise :
	x' = x cos f - y sin f
	y' = y cos f + x sin f
	*/

	//rotate endPoint & all interpolatedPoints offsetAngle degrees around startingPoint counterclockwise
	float3 nPoint;//used for saving point info
	nPoint.x = hairPoints[tid].endPoint.x;//x is 'same' on all points of hair, i.e when there is another wind that changes x direction this will POSSIBLY make things wrong
	nPoint.y = (hairPoints[tid].endPoint.y - hairPoints[tid].startPoint.y)*cosf(offsetAngleInRad) + (hairPoints[tid].endPoint.z - hairPoints[tid].startPoint.z)*sinf(offsetAngleInRad) +
		hairPoints[tid].startPoint.y;
	nPoint.z = (hairPoints[tid].endPoint.z - hairPoints[tid].startPoint.z)*cosf(offsetAngleInRad) - (hairPoints[tid].endPoint.y - hairPoints[tid].startPoint.y)*sinf(offsetAngleInRad) +
		hairPoints[tid].startPoint.z;
	hairPoints[tid].endPoint = nPoint;
	for (int i = 0; i < hairPoints[tid].interpolatedPointSize; i++)
	{
		nPoint.x = hairPoints[tid].interpolatedPoints[i].x;
		nPoint.y = (hairPoints[tid].interpolatedPoints[i].y - hairPoints[tid].startPoint.y)*cosf(offsetAngleInRad) + (hairPoints[tid].interpolatedPoints[i].z - hairPoints[tid].startPoint.z)*sinf(offsetAngleInRad) +
			hairPoints[tid].startPoint.y;
		nPoint.z = (hairPoints[tid].interpolatedPoints[i].z - hairPoints[tid].startPoint.z)*cosf(offsetAngleInRad) - (hairPoints[tid].interpolatedPoints[i].y - hairPoints[tid].startPoint.y)*sinf(offsetAngleInRad) +
			hairPoints[tid].startPoint.z;
		hairPoints[tid].interpolatedPoints[i] = nPoint;
	}

	//call collision detection @ ApplyWind
	
	//free
	free(nNonSmoothInterpolatedPoints);
	free(nSmoothInterpolatedPoints);
}


//gets hairPoints
//receives head radius from oHeadRadius
//if there is a collision between any point on hair(except startingPoint) & head, rotates the hair out of the head in +Y direction, does NOT CHANGES x point of the hair
//returns void, changes hair points' positions that are inside the head
__global__ void CollisionDetectionV0(hair* hairPoints)
{
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	//calculate minorLength which is the length of the hypothenuse of the O'-startingPoint.y-StartingPoint.z (O' is the center of the smaller circle which is on the same X point as rest of the points on this hair strand)
	float3 minorCenter;//center of the minor circle that passes through startPoint 
	minorCenter.x = hairPoints[tid].startPoint.x;
	minorCenter.y = 0;
	minorCenter.z = 0;
	float minorLength = length(hairPoints[tid].startPoint - minorCenter);
	float startingGammaRad = atan2f(hairPoints[tid].startPoint.y, hairPoints[tid].startPoint.z); //the angle(in radians) between starting point & +Z axis counterclockwise
																					   //we want the hair points(except startingpoint) to be 0.1f away from the head, simply push the hair 0.1f up from the head which is approx. 0.1f away from the origin

																					   //check the interpolatedPoints
	for (int i = 0; i < hairPoints[tid].interpolatedPointSize; i++)
	{
		//collision detected
		if (length(hairPoints[tid].interpolatedPoints[i] - minorCenter) < minorLength + 0.1f)//remember that we wanted the hairPoints to be at an offset of 0.1f
		{
			//how to: 
			//find a newPoint that is rotated out the head(vertically)
			//move all points (from hairPoints[tid].interpolatedPoints[i+1] to endPoint) to new positions while keeping the distance&angle between j'th & j-1'th point same
			//set the hairPoints[tid].interpolatedPoints[i] as new point


			//for more explanations on degrees etc. check the drawings 


			float n;//distance between 2 consecutive points on a hair strand
			float gammaRad;//the angle(in radians) between previous point & +Z axis counterclockwise
			float alphaPrimeRad;//the angle(in radians) between newPoint & previous point @origin counterclockwise

			if (i == 0)//if first interpolatedPoint is inside head
			{
				n = length(hairPoints[tid].interpolatedPoints[0] - hairPoints[tid].startPoint);

				//first edge is minorLength --> distance between startPoint&minorCenter, is always |minorLength|
				//second edge is minorLength+0.1f
				//third edge is n
				alphaPrimeRad = acosf((minorLength*minorLength + (minorLength + 0.1f)*(minorLength + 0.1f) - n * n) /
					(2 * minorLength*(minorLength + 0.1f)));
				if (isnan(alphaPrimeRad))alphaPrimeRad = PI;

				gammaRad = atan2f(hairPoints[tid].startPoint.y, hairPoints[tid].startPoint.z);
			}
			else//non first interpolatedPoint is inside head
			{
				n = length(hairPoints[tid].interpolatedPoints[i] - hairPoints[tid].interpolatedPoints[i - 1]);

				//first edge is prev distance=length(previous point- minorcenter), which is sometimes equal to minorLength+0.1f
				//second edge is minorLength+0.1f
				//third edge is n
				float prevDistance = length(hairPoints[tid].interpolatedPoints[i - 1] - minorCenter);

				alphaPrimeRad = acosf((prevDistance*prevDistance + (minorLength + 0.1f)*(minorLength + 0.1f) - n * n) /
					(2 * prevDistance * (minorLength + 0.1f)));
				if (isnan(alphaPrimeRad))  alphaPrimeRad = PI;

				gammaRad = atan2f(hairPoints[tid].interpolatedPoints[i - 1].y, hairPoints[tid].interpolatedPoints[i - 1].z);
			}

			float3 prevPoint = (i > 0 ? hairPoints[tid].interpolatedPoints[i - 1] : hairPoints[tid].startPoint);//the point that comes before this point
			float3 newPoint;
			newPoint.x = hairPoints[tid].interpolatedPoints[i].x;
			if ((prevPoint.y - n)*(prevPoint.y - n) + prevPoint.z*prevPoint.z>(minorLength + 0.1f)*(minorLength + 0.1f))//the hair falls down
			{
				newPoint.y = prevPoint.y - n;
				newPoint.z = prevPoint.z;
			}
			else//the hair follows curvature of the head
			{
				newPoint.y = (minorLength + 0.1f)*(sinf(alphaPrimeRad + gammaRad));
				newPoint.z = (minorLength + 0.1f)*(cosf(alphaPrimeRad + gammaRad));
			}

			//set rest of the points -- keep the distance & degrees between 2 consecutive points same
			//find vectors to add the previous point, which will give us the new points,
			float3 *moveVectors;//stores data on change of coordinates in consecutive points on hair, including endPoint
			moveVectors = (float3*)malloc(sizeof(float3)*(hairPoints[tid].interpolatedPointSize - i));
			for (int j = i + 1; j < hairPoints[tid].interpolatedPointSize; j++)
			{
				moveVectors[j - i - 1] = hairPoints[tid].interpolatedPoints[j] - hairPoints[tid].interpolatedPoints[j - 1];
			}
			moveVectors[hairPoints[tid].interpolatedPointSize - i - 1] = hairPoints[tid].endPoint - hairPoints[tid].interpolatedPoints[hairPoints[tid].interpolatedPointSize - 1];//set vector between end point - last interpolated point

			//set this point as new point
			hairPoints[tid].interpolatedPoints[i] = newPoint;
			//add respective moveVectors to rest of the interpolatedPoints
			for (int j = i + 1; j < hairPoints[tid].interpolatedPointSize; j++) hairPoints[tid].interpolatedPoints[j] = hairPoints[tid].interpolatedPoints[j - 1] + moveVectors[j - i - 1];

			//add moveVectors[-1] to endPoint
			hairPoints[tid].endPoint = hairPoints[tid].interpolatedPoints[hairPoints[tid].interpolatedPointSize - 1] + moveVectors[hairPoints[tid].interpolatedPointSize - i - 1];

			free(moveVectors);
		}
	}
	//check the endPoint
	if (length(hairPoints[tid].endPoint - minorCenter) < minorLength + 0.1f)
	{

		float n;//distance between 2 consecutive points on a hair strand
		float gammaRad;//the angle(in radians) between previous point & +Z axis counterclockwise
		float alphaPrimeRad;//the angle(in radians) between newPoint & +Z axis counterclockwise

							//calculate gammaRad & alphaPrimeRad
		if (hairPoints[tid].interpolatedPointSize > 0)//when there is at least 1 interpolatedPoint
		{
			n = length(hairPoints[tid].endPoint - hairPoints[tid].interpolatedPoints[hairPoints[tid].interpolatedPointSize - 1]);

			gammaRad = atan2f(hairPoints[tid].interpolatedPoints[hairPoints[tid].interpolatedPointSize - 1].y, hairPoints[tid].interpolatedPoints[hairPoints[tid].interpolatedPointSize - 1].z);

			//first edge is prev distance=length(previous point- minorcenter), which is sometimes equal to minorLength+0.1f
			//second edge is minorLength+0.1f
			//third edge is n
			float prevDistance = length(hairPoints[tid].interpolatedPoints[hairPoints[tid].interpolatedPointSize - 1] - minorCenter);

			alphaPrimeRad = acosf((prevDistance*prevDistance + (minorLength + 0.1f)*(minorLength + 0.1f) - n * n) /
				(2 * prevDistance * (minorLength + 0.1f)));
			if (isnan(alphaPrimeRad)) alphaPrimeRad = PI;//acosf can return nan
		}
		else//no interpolation --> previous point is starting point
		{
			n = length(hairPoints[tid].endPoint - hairPoints[tid].startPoint);
			gammaRad = atan2f(hairPoints[tid].startPoint.y, hairPoints[tid].startPoint.z);
			//first edge is minorlength
			//second edge is minorLength+0.1f
			//third edge is n
			alphaPrimeRad = acosf((2 * minorLength*(minorLength + 0.1f) - n * n) /
				(2 * minorLength*(minorLength + 0.1f)));
			if (isnan(alphaPrimeRad)) alphaPrimeRad = PI;//acosf can return nan
		}

		float3 prevPoint = (hairPoints[tid].interpolatedPointSize > 0 ? hairPoints[tid].interpolatedPoints[hairPoints[tid].interpolatedPointSize - 1] : hairPoints[tid].startPoint);//the point that comes before end point
		//rotate endPoint out of the head
		if ((prevPoint.y - n)*(prevPoint.y - n) + prevPoint.z*prevPoint.z>(minorLength + 0.1f)*(minorLength + 0.1f))//hair falls down
		{
			hairPoints[tid].endPoint.y = prevPoint.y - n;
			hairPoints[tid].endPoint.z = prevPoint.z;
		}
		else//hair follows curvature of the head
		{
			hairPoints[tid].endPoint.y = (minorLength + 0.1f)*(sinf(alphaPrimeRad + gammaRad));
			hairPoints[tid].endPoint.z = (minorLength + 0.1f)*(cosf(alphaPrimeRad + gammaRad));
		}
	}
}

//frees hairPoints_device
void FreeAllGPU(hair *hairPoints)
{
	//for (int i = 0; i < oHairCount_kernel; i++) free((hairPoints[i].interpolatedPoints));// interpolated points are already free'd with h_data.interpoalted point since they point to the same point on memory
	gpuErrchk(cudaFree(hairPoints));
}
