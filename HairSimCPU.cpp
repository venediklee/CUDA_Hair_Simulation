#include "HairSimCPU.h"

#define PI 3.14159265358979f

//these are original values, DO NOT FORGET TO SAVE THEM--especially hair count, head radius, hair length
hair *oHairPoints;
int oHairCount;
float oHeadRadius;
float oHairLength;

//used for setting up the scene
//head position is assumed to be at 0
//headRadius is radius of the head, which is spherical
//hairPoints stores data of all hairs
//hairLength is length of each hairstrand
//TODO (performance decrease & memory improvement) no need to save original interpolatedPoints as they are related with hairPointsB etc.
void SaveOriginalSettings(float headRadius, int hairCount, hair* hairPoints, float hairLength)
{
	if (hairPoints == NULL)
	{
		throw std::invalid_argument("hairPoints is not set while calling Settings method");
		exit(-1);
	}
	
	//free old values
	free(oHairPoints);

	//allocate memory
	oHairPoints = (hair*)malloc(sizeof(hair)*hairCount);

	//set the values
	oHeadRadius = headRadius;
	oHairCount = hairCount;
	oHairLength = hairLength;
	memcpy(oHairPoints, hairPoints, sizeof(hair)*hairCount);
	for (int i = 0; i < hairCount; i++)
	{
		//free & malloc & cpy interpolatedPoints
		//free(oHairPoints[i].interpolatedPoints);
		oHairPoints[i].interpolatedPoints=(float3*)malloc(sizeof(float3)* hairPoints[i].interpolatedPointSize);
		memcpy(oHairPoints[i].interpolatedPoints, hairPoints[i].interpolatedPoints, sizeof(float3)*hairPoints[i].interpolatedPointSize);
	}
}

//gets hairCount, hairRadius &  hair *hairPoints
// sets hairPoints.startingPoint(s) to appropriate values
//hair only comes out in 3/4th of head -- look at the bottom of this script for a visualization on where the hair comes out etc.
//returns void, changes hairPoints.startPoint(s)
void HairPointASetter(int hairCount, float headRadius, hair *hairPoints)
{
	if (sizeof(hairPoints) == 0)
	{
		std::cout << "hairPoints haven't been malloc'd when calling HairPointASetter" << std::endl;
		throw std::invalid_argument("hairPoints haven't been malloc'd when calling HairPointASetter");
		exit(-1);
	}

	float3 v;
	float remainingDist;// used for calculating positions of y and x correctly
	srand(static_cast <unsigned> (time(0))); //seed random
	for (int i = hairCount-1; i >= 0; i--)
	{
		//z can be [-headRadius,headRadius]
		v.z = -headRadius + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (2 * headRadius)));
		remainingDist = sqrtf(headRadius * headRadius - v.z*v.z);

		if (v.z > 0)// y cant be negative if z is positive ##look at the bottom of the script for explanation
		{
			if (remainingDist == 0)v.y = 0;//division by 0 prevention
			else v.y = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (2 * remainingDist))); //y can be [0,remainingDist]
		}
		else
		{
			//y can be [-remainingDist,remainingDist]
			if (remainingDist == 0) v.y = 0;//division by 0 prevention
			else v.y = -remainingDist +  static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (2 * remainingDist))); 
		}
		remainingDist = sqrtf(headRadius * headRadius - v.z*v.z - v.y*v.y);

		//x can be -remainingDist or remainingDist
		if (remainingDist == 0) v.x = 0;//divison by 0 prevention
		else v.x = remainingDist * (static_cast<float>(rand()) > (static_cast<float> (RAND_MAX) / 2.0f) ? 1 : -1);

		if (isnan(v.x) || isnan(v.y)|| isnan(v.z) || headRadius*headRadius - length(v) < 0)//recalculate
		{
			i++;
			continue;
		}

		//set starting point
		hairPoints[i].startPoint = v;
	}
}

//gets hairCount,hair *hairPoints & interpolatedPointSize
//mallocs interpolatedPoints & sets the interpolatedPointSize for each hair
//returns void, mallocs hairPoints.interpolatedPoints & changes hairPoints.interpolatedPointSize
void HairInterpolationSetter(int hairCount, hair* hairPoints, int interpolatedPointSize)
{
	for (int i = hairCount-1; i >=0 ; i--)
	{
		hairPoints[i].interpolatedPointSize = interpolatedPointSize;
		hairPoints[i].interpolatedPoints = (float3*)malloc(sizeof(float3)*interpolatedPointSize);
	}
}


//gets hairCount, hairRadius, hairLength &  hairPoints
//hairLength must be at least 1.45 times the headRadius
// sets  hairPoints.endPoint(s) to appropriate values
	//also creates and sets interpolated points if there are any
// offset for hair strands points are 0.1f, but this is costumizable
void HairPointBSetter(int hairCount, float headRadius, float hairLength, hair *hairPoints)
{
	// hair grows to the back(priority) & down the head, i.e hairPointsB[i].z < hairPointsA[i].z
	// first set the end point, then check for collision detection
		//this makes it easier to set the hair, but some parts of the hair strand might be inside the hair
			// to mitigate this and give the hair volume: inside collision detection, push the hair with a random offset from a small interval.->i chose 0.1f
			// we can also set interpolated points to make the hair curve around the head.

	

	if (hairPoints == NULL)
	{
		throw std::invalid_argument("hairPoints is not set when calling HairPointBSetter");
		exit(-1);
	}
	
	// start each end point at a 45 degree angle above and behind the starting point(do not change the x position)
		//set the interpolatedPoints
	//pull each strand downwards until they collide with head or reach the bottom point
		//this sets the hair correctly
	//TODO additional (performance improvement): push hairs that are below origin down, dont let them enter for loop, they wont need it
	float3 v;//used for setting & manipulating points
	for (int i = hairCount-1; i >= 0; i--)
	{
		//set end point
		v.x = hairPoints[i].startPoint.x;
		v.y = hairPoints[i].startPoint.y + hairLength / sqrtf(2.0f);//put the hair 45 degree up
		v.z = hairPoints[i].startPoint.z - hairLength / sqrtf(2.0f);//put the hair 45 degree back

		hairPoints[i].endPoint = v;

		if (hairPoints[i].interpolatedPoints == NULL && hairPoints[i].interpolatedPointSize > 0)
		{
			std::cout<< ("hairPoints[????].interpolatedPoints is not malloc'd when calling HairPointBSetter")<<std::endl;
			throw std::invalid_argument("hairPoints[????].interpolatedPoints is not malloc'd when calling HairPointBSetter");
			exit(-1);
		}

		//set interpolatedPoints (they are already malloc'd with HairInterpolationSetter method)
		for (int j = hairPoints[i].interpolatedPointSize-1; j >= 0; j--)
		{
			
			//x is the same on all points on the hair strand when there is no wind

			v.y = (hairPoints[i].endPoint.y - hairPoints[i].startPoint.y)*((float)j+1)
				/ ((float)hairPoints[i].interpolatedPointSize + 1) + hairPoints[i].startPoint.y;

			v.z = (hairPoints[i].endPoint.z - hairPoints[i].startPoint.z)*((float)j+1)
				/ ((float)hairPoints[i].interpolatedPointSize + 1) + hairPoints[i].startPoint.z;

			hairPoints[i].interpolatedPoints[j] = v;
		}
		
		//all hairs are now pointing above & back from the head
			//problem: an interpolated point is inside the head
			//solution1: instead of setting the hairs to above & back of the head, set hairs above the origin to above first
				//then push them back, then push them down. hairs below origin only goes down anyways
			//solution2: check for collision detection
				//solution2 is better
		//then drop(slerp) them until all points of head ( touch with head or is vertical )
	}
	
	//do collision detection on all hairs
	for (int i = hairCount-1; i >= 0; i--)
	{
		CollisionDetection(hairPoints+i);
	}

	//drop the hair
	//how to: slerp the hair downwards(-y) (or rotate hair counterclockwise from +Z axis --assuming eyes are looking right(+Z axis)--) 
		//until the first interpolatedPoint or endPoint gets NEAR the head
		//it is almost same as wind effects etc.
	for (int i = hairCount-1; i >= 0; i--)
	{
		/* things to note:
		*	head's center is origin
		*	the angles are on YZ plane
		*	the angles are calculated with respect to +z axis(counterclockwise)
		*	the angle between startingPoint & +z axis(counterclockwise) is gamma
		*	############## so gamma(radians)= atan2f(startingPoint.y, startingPoint.z) ##############
		*	cpp sin/cos functions are done in radian
		*	sin(30)=sin(pi/6) etc.
		*	n is length of each hair partition, i.e if there is 1 interpolatedPoint 2n=hairLength
		*/

		/* The drop algorithm step-by-step
		*
		*	first step: if there is interpolatedPoint, set first interpolatedPoint
		*				else set the endPoint then finish it
		*	second step: set the rest of the interpolatedPoints & endPoint, then finish it
		*
		*---------------------------------------
		***********************first step: 
		*	the rotation angle of the newPoint is alpha
		*	alpha is the angle at origin in the triangle (origin-startingPoint)&(origin-newPoint)&(newPoint-startingPoint), alpha looks at the first edge
		*	First edge: is the distance between (newPoint-startingPoint) 
		*				the length(distance) does not change from the (oldPoint-startingPoint) & (newPoint-startingPoint), i.e it remains n
		*	Second edge: is the distance between (origin-startingPoint)
		*				the length is headRadius
		*	Third edge: is the distance between (origin-newPoint)
		*				the length is headRadius+0.1f ---> +0.1f is to give the hair some volume but this can be changed or costumized
		*	############## so alpha(radians)= acosf( (SecondEdge^2 + ThirdEdge^2 - FirstEdge^2) / ( 2*SecondEdge*ThirdEdge ) ############## 
		*	############## coordinates of newPoint is: (gamma+alpha >= PI) ? (previous point.x, previous point.z-n) 
		*																		: ( (headRadius+0.1f) * ( sinf(gamma+alpha) , cosf(gamma+alpha) ) )
		*
		*
		***********************second step: 
		*	the rotation angle of the newPoint2 is beta
		*	beta is the angle at origin in the triangle (origin-newPoint)&(origin-newPoint2)&(newPonit2-newPoint), beta looks at the first edge
		*	First edge: is the distance between (newPoint2-newPoint)
		*				the length(distance) does not change from the (oldPoint2-oldPoint) & (newPOint2-newPoint), i.e it remains n
		*	Second edge: is the distance between(origin-newPoint)
		*				the length is headRadius+0.1f
		*	Third edge: is the distance between (origin-newPoint2)
		*				the length is headRadius+0.1f
		*	############## so beta(radians)= acosf( ( 2*(headRadius+0.1f)^2 - n^2 ) / 2*(headRadius+0.1f)^2 ) ##############
		*	############## coordinates of newPoint2 is:(gamma+alpha+beta*iterationStep >= PI) ? (previous point.x, previous point.z-n) 
		*																					: ( (headRadius+0.1f) * ( sinf(gamma+alpha+beta*iterationStep) , cosf(gamma+alpha+beta*iterationStep) ) )
		*	REPEAT this step until you set each interpolatedPoint & endPoint
		*	note that the edges will change accordingly, but the angle & coordinate calculations are correct
		*/

		int interPSize = hairPoints[i].interpolatedPointSize;//hairPoints[i].interpolatedPointSize

		//calculate gamma(angle of startingPoint in radians)
		float gammaRad = atan2f(hairPoints[i].startPoint.y, hairPoints[i].startPoint.z);

		//calculate minorLength which is the length of the hypothenuse of the O'-startingPoint.y-StartingPoint.z (O' is the center of the smaller circle which is on the same X point as rest of the points on this hair strand)
		float3 minorCenter;
		minorCenter.x = hairPoints[i].startPoint.x;
		minorCenter.y = 0;
		minorCenter.z = 0;
		float minorLength = length(hairPoints[i].startPoint - minorCenter);

		if (interPSize > 0)//set interpolatedPoints & endPoint
		{
			//set the first interpolatedPoint according to step1

			//calculate n (size of hairLength partitions)
			float n = hairLength / ((float)(interPSize + 1));

			//calculate alpha (the first points angle in radians)
			//first edge is minorLength --> distnce between startPoint&minorCenter
			//second edge is minorLength+0.1f
			//third edge is n
			float alphaRad = acosf((minorLength*minorLength + (minorLength + 0.1f)*(minorLength + 0.1f) - n * n) /
				(2 * minorLength*(minorLength + 0.1f)));

			if (isnan(alphaRad)) alphaRad = PI;//acosf returns nan if inside of acosf gets values smaller than -1
			
			//the point should go directly down instead of down& towards the head if the hair can fall through the head with no collision
			if (((hairPoints[i].startPoint.y-n)*(hairPoints[i].startPoint.y-n)+hairPoints[i].startPoint.z*hairPoints[i].startPoint.z)>(minorLength+0.1f)*(minorLength+0.1f))
			{
				hairPoints[i].interpolatedPoints[0].y = hairPoints[i].startPoint.y - n;
				hairPoints[i].interpolatedPoints[0].z = hairPoints[i].startPoint.z;
			}
			else//the point follows the curveture of the head
			{
				hairPoints[i].interpolatedPoints[0].y = (minorLength + 0.1f) * sinf(gammaRad + alphaRad);
				hairPoints[i].interpolatedPoints[0].z = (minorLength + 0.1f) * cosf(gammaRad + alphaRad);
			}

			//calculate beta (the consecutive points angles in radian)
			//first edge is minorLength+0.1f
			//second edge is minorLength+0.1f
			//third edge is n
			float betaRad = acosf((2 * (minorLength+0.1f)*(minorLength +0.1f)- n * n) / (2 * (minorLength + 0.1f) * (minorLength+ 0.1f)));
			if (isnan(betaRad)) betaRad = PI;
			float betaRadStep = betaRad;
			for (int j = 1; j < interPSize ; j++)//set the rest of the interpolatedPoints according to step2
			{
				//the point should go directly down instead of down& towards the head if the hair can fall through the head with no collision
				if (((hairPoints[i].interpolatedPoints[j-1].y-n)*(hairPoints[i].interpolatedPoints[j - 1].y - n)+
						hairPoints[i].interpolatedPoints[j-1].z*hairPoints[i].interpolatedPoints[j-1].z>(minorLength+0.1f)*(minorLength+0.1f)))
				{
					hairPoints[i].interpolatedPoints[j].y = hairPoints[i].interpolatedPoints[j - 1].y - n;
					hairPoints[i].interpolatedPoints[j].z = hairPoints[i].interpolatedPoints[j - 1].z;
				}
				else//the point follows the curveture of the head
				{
					hairPoints[i].interpolatedPoints[j].y = (minorLength + 0.1f) * sinf(gammaRad + alphaRad + betaRad);
					hairPoints[i].interpolatedPoints[j].z = (minorLength + 0.1f) * cosf(gammaRad + alphaRad + betaRad);
				}
				//just increase betaRad as much as first betaRad instead of multiplying at each step 
				betaRad += betaRadStep;//betaRad becomes the next points betaRad, at the end it becomes the betaRad of the endPoint
			}

			//set the endPoint according to step2
			//the point should go directly down instead of down& towards the head if the hair can fall through the head with no collision
			if (((hairPoints[i].interpolatedPoints[interPSize - 1].y - n)*(hairPoints[i].interpolatedPoints[interPSize - 1].y - n) +
					hairPoints[i].interpolatedPoints[interPSize - 1].z*hairPoints[i].interpolatedPoints[interPSize - 1].z>(minorLength + 0.1f)*(minorLength + 0.1f)))
			{
				hairPoints[i].endPoint.y = hairPoints[i].interpolatedPoints[interPSize - 1].y - n;
				hairPoints[i].endPoint.z = hairPoints[i].interpolatedPoints[interPSize - 1].z;
			}
			else//the point follows the curveture of the head
			{
				hairPoints[i].endPoint.y = (minorLength + 0.1f) * sinf(gammaRad + alphaRad + betaRad);
				hairPoints[i].endPoint.z = (minorLength + 0.1f) * cosf(gammaRad + alphaRad + betaRad);
			}
		}
		else//just(only) set the endPoint according to step1
		{
			//calculate alpha (the first points angle in radians)
			float alphaRad = acosf((minorLength*minorLength + (minorLength + 0.1f)*(minorLength + 0.1f) - hairLength * hairLength) /
				(2 * minorLength*(minorLength + 0.1f)));
			if (isnan(alphaRad)) alphaRad = PI;// acosf returns nan if inside of acosf gets values smaller than -1

			//the point should go directly down instead of down& towards the head if the hair can go thorugh the head with no collisions
			if ((hairPoints[i].startPoint.y - hairLength)*(hairPoints[i].startPoint.y - hairLength) +
				hairPoints[i].startPoint.z*hairPoints[i].startPoint.z > (minorLength + 0.1f)*(minorLength + 0.1f))
			{
				hairPoints[i].endPoint.y = hairPoints[i].startPoint.y - hairLength;
				hairPoints[i].endPoint.z = hairPoints[i].startPoint.z;
			}
			else//the point follows the curveture of the head
			{
				hairPoints[i].endPoint.y = (minorLength + 0.1f) * sinf(gammaRad + alphaRad);
				hairPoints[i].endPoint.z = (minorLength + 0.1f) * cosf(gammaRad + alphaRad);
			}
		}
		
	}
	for (int i = 0; i < hairCount; i++) CollisionDetection(hairPoints + i);//collision detection just in case something went wrong
}




//gets thrStruct which includes tid,hairPoints, wind(strength,axis,axisSign), smoothing=5
//hair points is pointer to hair data
//wind origin is a char {+-X, +-Y, +-Z}, denotes the origin point of the wind, i.e wind blows from wind origin towards head
//wind strength is a float used for determining the amount each individual hair moves-rotates 
//smoothing is the amount of steps required for each hair strand to reach peak position
//recieves hairCount, headRadius from global variable(s) oHairCount, oHeadRadius
//TODO additional feature can use this routine multiple times without stopping all winds
//returns void*(must), changes the direction-rotation of hair OVER 'TIME' UP TO PEAK POSITION depending on smoothing by calling startWind method on CPU
void ApplyWind(void *thrStr)
{
	threadStruct *th = (threadStruct*)thrStr;
	int tid = th->tid;
	hair *hairPoints = th->hairPoints;
	wind w = th->w;
	int smoothing = th->smoothing;
	int hairPartitionSize = th->hairPartitionSize;

	if (w.strength == 0)
	{
		return;
	}
	if (hairPoints == NULL)
	{
		throw std::invalid_argument("hairPoints is not malloc'd when calling ApplyWind");
		exit(-1);
	}
	if ((w.axis != 'X' && w.axis != 'Y' && w.axis != 'Z') || (w.axisSign != '+' && w.axisSign != '-'))
	{
		throw std::invalid_argument("wind is not set correctly when calling ApplyWind");
		exit(-1);
	}
	if (hairPartitionSize <= 0)
	{
		throw std::invalid_argument("hairPartitionSize is not set correctly when calling ApplyWind");
		exit(-1);
	}

	//change the position of the hair that is the same as wind axis
	/*	how to (with an example on Z axis):
	*	if the wind is blowing from +Z(eyes are looking at +Z) increase the y value of all hair points(except startingPoint) 
	*															while decreasing Z value(the total distance between points must remain same)
	*	############## max Y increase: = (startingPoint.y-point.y)*effectiveStrength
	*	effectiveStrength is based on the distance of the hair(startinPoint) to the origin of the hair, since the head blocks some of the wind,
	*		meaning that the closest hairs to the wind have highest effectiveStrength
	*			effectiveStrength should not be too close to 0, so make it [0.5,1]*windStrength
	*	############## so effectiveStrength=(0.75f+(startingPoint.z)/(4*headRadius))*windStrength
	*	so no point can go above the Y position of startingPoint of that individual hair Strand
	*	
	*
	*	############################ setting points ############################
	*	#	NOTE: new variables start with n-- i.e nEndPoint, nHairStrand, nNonsmoothInterpolatedPoints,nSmoothInterpolatedPoints;
	*	#	calculate nEndPoint
	*	#	############## nEndPoint=(startingPoint.y-endPoint.y)*effectiveStrength
	*	#	nHairStrand is the line between nEndPoint & startingPoint
	*	#	calculate nNonsmoothInterpolatedPoints which are equally partitioned points on nHairStrand(first element is closest to startingPoint)
	*	#	############## nNonsmoothInterpolatedPoints[i]= (nEndPoint-startinPoint)*(i+1)/(interpolatedPointSize+1) + startingPoint --> i=[0,interpolatedPointSize-1]
	*	#	set the angle(rotate, from +Z to +Y counterclockwise) of nSmoothInterpolatedPoints[i] as		-->note that there is no .angle attribute
	*	#	::		(nNonsmoothInterpolatedPoints[i].angle-interpolatedPoints[i].angle)*(i+1)/(interpolatedPointSize+1)+interpolatedPoints[i].angle
	*	#	:: set the coordinates of nSmoothInterpolatedPoints[i] based on distance of interpolatedPoints[i] to startingPoint & calculated angle 
	*	#	set the interpolatedPoints[0] to nSmoothInterpolatedPoints[0]
	*	#	for each interpolatedPoints[1,interpolatedPointSize-1]
	*	#	:: find the NORMALIZED vector from interpolatedPoints[i-1] to nSmoothInterpolatedPoints[i] called nVector
	*	#	:: set interpolatedPoints[i] to interpolatedPoints[i-1] + nVector * hairLength/(intepolatedPointSize+1)
	*	#	set the endPoint to interpolatedPoints[interpolatedPointSize-1]+ nVector* hairLength/(intepolatedPointSize+1) --> nVector is from last interpolatedPoint to nEndPoint
	*	
	*
	*	NOTE: at this point the hair points are set, but they look really bad & are not correct(i.e endPoint is not at the same height as nEndPoint), time to correct them
	*	############################ correcting points ############################
	*	#	find the nAngle between endPoint-startingPoint & nEndPoint-startingPoint
	*	#	rotate the endPoint & all interpolatedPoints nAngle degrees towards nEndPoint-startingPoint line
	*	
	*	############################ collision detection ############################
	*	#	for all points except startingPoint(s) check if the distance to origin is less than headRadius
	*	#	if so
	*	#	:: calculate the angle needed to push interpolatedPoint[0] outside the head and rotate each point that amount
	*
	*
	*	############################ now repeat setting & correcting & collision detection steps smoothing amount of times ############################
	*	#	this can easily be done with following code
	*	#	:: for each smoothingStep --> smoothingStep=[1,smoothing]
	*	#		:: multiply nEndpoint with smoothingStep/smoothing
	*/



	/*	how to(general notes)
	*	DO NOT APPLY WIND IN Y DIRECTION IF THERE WERE NO PREVIOUS WINDS APPLIED
	*	if wind is in Z direction change y,z points(z changes depending on wind, y changes depending on z)
	*	if wind is in X direction change x,y points(x changes depending on wind, y changes depending on x)
	*	if wind is in Y direction change y,z points(y changes depending on wind, z changes depending on y)
	*	you can also check the drawing i included
	*/
	
	for (int j = 0; j<smoothing ; j++)//at each smoothing step
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
			for (int i = hairPartitionSize * (tid + 1) - 1; i >= hairPartitionSize * tid; i--)//for each hair (this thread is responsible for)
			{
				StartWindZ(hairPoints+i, w.axisSign, smoothedStrength);
			}
		}
		
		//we can sync threads here to prevent some hairs reaching peak position first etc., but it is not necessary & decreases performance
	}

}



//!!!gets hairPoint+i, wind axis sign, wind strength
//receives hair length from oHairLength--saved@ SaveOriginalSettings
//hair point is pointer to ONE hair strand data
//wind axis sign determines the clockwise-counterclokwise'ness of hair rotation
//wind strength determines the angle of hair rotation
//returns void, changes positions of ONE hair strand based on the type of wind
//other versions are StartWindY, StartWindX
void StartWindZ(hair *hairPoint, char sign, float strength)
{
	//look at applywind() for more info

	//change y,z points, x is always same
	//if sign is +, z point gets decreased vice versa
	//find effective wind strength--same on all points of the hair
	float effectiveStrength = (0.75f + (hairPoint->startPoint.z) / (4 * oHeadRadius))*strength;

	//calculate nEndPoint
	float3 nEndPoint;
	nEndPoint.x = hairPoint->startPoint.x;
	nEndPoint.y = (hairPoint->startPoint.y - hairPoint->endPoint.y)*effectiveStrength + hairPoint->endPoint.y;
	float Zdist = sqrtf(oHairLength*oHairLength - (nEndPoint.y - hairPoint->startPoint.y)*(nEndPoint.y - hairPoint->startPoint.y));

	if (sign == '+')
	{
		nEndPoint.z = hairPoint->startPoint.z - Zdist;
	}
	else//sign == '-'
	{
		nEndPoint.z = hairPoint->startPoint.z + Zdist;
	}

	float3 * nNonSmoothInterpolatedPoints, *nSmoothInterpolatedPoints;
	nNonSmoothInterpolatedPoints = (float3*)malloc(sizeof(float3)*hairPoint->interpolatedPointSize);
	nSmoothInterpolatedPoints = (float3*)malloc(sizeof(float3)*hairPoint->interpolatedPointSize);
	//calculate nNonSmoothInterpolatedPoints for each interpolated point, then set nSmoothInterpolatedPoints
	for (int i = 0; i < hairPoint->interpolatedPointSize; i++)
	{
		//smoothPoint calculations
		nNonSmoothInterpolatedPoints[i] = (nEndPoint - hairPoint->startPoint)*((float)(i + 1)) / ((float)(hairPoint->interpolatedPointSize + 1)) + hairPoint->startPoint;
		float nRad = atan2f(nNonSmoothInterpolatedPoints[i].y - hairPoint->startPoint.y, nNonSmoothInterpolatedPoints[i].z - hairPoint->startPoint.z);//the angle(in radians) of nonSmoothInterpolatedPoint[i]-startingPoint from the +Z axis(counterclockwise)
		float rad = atan2f(hairPoint->interpolatedPoints[i].y - hairPoint->startPoint.y, hairPoint->interpolatedPoints[i].z - hairPoint->startPoint.z);//the angle(in radians) of interpolatedPoints[i]-startingPoint from +Z axis(counterclockwise)
		float YZdistToStart = sqrtf((nNonSmoothInterpolatedPoints[i].y - hairPoint->startPoint.y)*(nNonSmoothInterpolatedPoints[i].y - hairPoint->startPoint.y)
			+ (nNonSmoothInterpolatedPoints[i].z - hairPoint->startPoint.z)*(nNonSmoothInterpolatedPoints[i].z - hairPoint->startPoint.z));
		
		nSmoothInterpolatedPoints[i].x = nNonSmoothInterpolatedPoints[i].x;
		nSmoothInterpolatedPoints[i].y = YZdistToStart * sinf((nRad - rad)*(i + 1) / (hairPoint->interpolatedPointSize+1) + rad) +hairPoint->startPoint.y;//equally divides the angle between nonSmoothInterpolatedPoints & interpolatedPoints, then sets the angle of smoothPoint[i] as i'th step between nonSmoothInterpolatedPoint& interpolatedPoints 
		nSmoothInterpolatedPoints[i].z = YZdistToStart * cosf((nRad - rad)*(i + 1) / (hairPoint->interpolatedPointSize + 1) + rad)+hairPoint->startPoint.z;
	}
	
	//move hair points
	float3 nVector;//used for finding vector from one point to another
	if(hairPoint->interpolatedPointSize>0) hairPoint->interpolatedPoints[0] = nSmoothInterpolatedPoints[0];
	for (int i = 1; i < hairPoint->interpolatedPointSize; i++)
	{
		//find NORMALIZED vector from interpoaltedPoints[i-1] to nSmoothInterpolatedPoints[i]
		nVector =normalize( nSmoothInterpolatedPoints[i] - hairPoint->interpolatedPoints[i - 1]);

		//set interpolatedPoints[i] to interpolatedPoints[i-1] + nVector * hairLength/(intepolatedPointSize+1)
		hairPoint->interpolatedPoints[i] = hairPoint->interpolatedPoints[i - 1] + (nVector / (float)(hairPoint->interpolatedPointSize + 1))*sqrtf(
			(nEndPoint.y - hairPoint->startPoint.y)*(nEndPoint.y - hairPoint->startPoint.y) + (nEndPoint.z - hairPoint->startPoint.z)*(nEndPoint.z - hairPoint->startPoint.z));
	}
	
	//set endPoint
	if (hairPoint->interpolatedPointSize > 0)
	{
		nVector = normalize(nEndPoint - hairPoint->interpolatedPoints[hairPoint->interpolatedPointSize - 1]);
		hairPoint->endPoint = hairPoint->interpolatedPoints[hairPoint->interpolatedPointSize - 1] + nVector / (float)(hairPoint->interpolatedPointSize + 1)*sqrtf(
			(nEndPoint.y - hairPoint->startPoint.y)*(nEndPoint.y - hairPoint->startPoint.y) + (nEndPoint.z - hairPoint->startPoint.z)*(nEndPoint.z - hairPoint->startPoint.z));
	}
	else
	{
		//no interpolation:=the hair will be a single line
		hairPoint->endPoint = nEndPoint;
	}

	
	
	//correct points
	float angleInRad = atan2f((hairPoint->endPoint - hairPoint->startPoint).y, (hairPoint->endPoint - hairPoint->startPoint).z);//angle between current endPoint& +Z axis counterclockwise
	float nAngleInRad = atan2f((nEndPoint - hairPoint->startPoint).y, (nEndPoint - hairPoint->startPoint).z);	//angle between nEndPoint& +Z axis counterclockwise

	float offsetAngleInRad = nAngleInRad - angleInRad;//rotate the hair this much counterclockwise
	/*	2D rotating of a point around origin counterclockwise :
		x' = x cos f - y sin f
		y' = y cos f + x sin f
	*/

	//rotate endPoint & all interpolatedPoints offsetAngle degrees around startingPoint counterclockwise
	float3 nPoint;//used for saving point info
	nPoint.x = hairPoint->endPoint.x;//x is 'same' on all points of hair, i.e when there is another wind that changes x direction this will POSSIBLY make things wrong
	nPoint.y = (hairPoint->endPoint.y - hairPoint->startPoint.y)*cosf(offsetAngleInRad) + (hairPoint->endPoint.z - hairPoint->startPoint.z)*sinf(offsetAngleInRad) +
		hairPoint->startPoint.y;
	nPoint.z= (hairPoint->endPoint.z - hairPoint->startPoint.z)*cosf(offsetAngleInRad) - (hairPoint->endPoint.y - hairPoint->startPoint.y)*sinf(offsetAngleInRad) +
		hairPoint->startPoint.z;
	hairPoint->endPoint = nPoint;
	for (int i = 0; i < hairPoint->interpolatedPointSize; i++)
	{
		nPoint.x = hairPoint->interpolatedPoints[i].x;
		nPoint.y = (hairPoint->interpolatedPoints[i].y - hairPoint->startPoint.y)*cosf(offsetAngleInRad) + (hairPoint->interpolatedPoints[i].z - hairPoint->startPoint.z)*sinf(offsetAngleInRad) +
			hairPoint->startPoint.y;
		nPoint.z = (hairPoint->interpolatedPoints[i].z - hairPoint->startPoint.z)*cosf(offsetAngleInRad) - (hairPoint->interpolatedPoints[i].y - hairPoint->startPoint.y)*sinf(offsetAngleInRad) +
			hairPoint->startPoint.z;
		hairPoint->interpolatedPoints[i] = nPoint;
	}
	
	
	CollisionDetection(hairPoint);
	
	//free
	free(nNonSmoothInterpolatedPoints);
	free(nSmoothInterpolatedPoints);
}

//TODO StartWindY not implemented yet
//check StartWindZ for explanations
void StartWindY(hair *hairPoint, char sign, float strength)
{
}

//TODO StartWindX not implemented yet
//check StartWindZ for explanations
void StartWindX(hair *hairPoint, char sign, float strength)
{
}

//NOTE that this function is not functioning like it says, since that would make the projects scope too large
//gets hairPoints, overTime=0.3f
//hair points is pointer to hair data
//overTime is the amount of time required for each hair strand to reach default position(stored in oHairPoints), default 0.3f
//returns void, changes all hair position-rotation to its original position stored in hair *oHairPoints OVER TIME
void StopAllWinds(hair *hairPoints,float overTime=0.3f)
{
	//TODO stop wind is not impelemented
	//just applying wind from +Y direction with full strength will suffice for now--may not be accurate/correct
	//create cpu threads
	const int CPUThreadCount = 16;
	int smoothing = 5;
	std::thread CPUthreads[CPUThreadCount];
	wind w;
	w.axis = 'Y';
	w.axisSign = '+';
	w.strength = 1;

	for (int i = 0; i < CPUThreadCount; i++)//start multithreaded functions
	{
		//create function attribute
		threadStruct thStruct;
		thStruct.hairPoints = hairPoints;
		thStruct.smoothing = smoothing;
		thStruct.tid = i;
		thStruct.w = w;
		thStruct.hairPartitionSize = oHairCount / CPUThreadCount;//equal partition is ensured
																//start cpu threads
		CPUthreads[i] = std::thread(ApplyWind, (void*)&thStruct);
	}

	//join threads
	for (int i = 0; i < CPUThreadCount; i++)
	{
		CPUthreads[i].join();
	}
}


//gets a hairPoint
//receives head radius from oHeadRadius
//if there is a collision between any point on hair(except startingPoint) & head, rotates the hair out of the head in +Y direction, does NOT CHANGES x point of the hair
//returns void, changes hair points' positions that are inside the head
void CollisionDetection(hair* hairPoint)
{
	//calculate minorLength which is the length of the hypothenuse of the O'-startingPoint.y-StartingPoint.z (O' is the center of the smaller circle which is on the same X point as rest of the points on this hair strand)
	float3 minorCenter;//center of the minor circle that passes through startPoint 
	minorCenter.x = hairPoint->startPoint.x;
	minorCenter.y = 0;
	minorCenter.z = 0;
	float minorLength = length(hairPoint->startPoint - minorCenter);
	float startingGammaRad = atan2f(hairPoint->startPoint.y, hairPoint->startPoint.z); //the angle(in radians) between starting point & +Z axis counterclockwise
	//we want the hair points(except startingpoint) to be 0.1f away from the head, simply push the hair 0.1f up from the head which is approx. 0.1f away from the origin
	
	//check the interpolatedPoints
	for (int i = 0; i < hairPoint->interpolatedPointSize; i++)
	{
		//collision detected
		if (length(hairPoint->interpolatedPoints[i]-minorCenter) < minorLength+ 0.1f)//remember that we wanted the hairPoints to be at an offset of 0.1f
		{
			//how to: 
			//find a newPoint that is rotated out the head(vertically)
			//move all points (from hairPoint->interpolatedPoints[i+1] to endPoint) to new positions while keeping the distance&angle between j'th & j-1'th point same
			//set the hairPoint->interpolatedPoints[i] as new point

			
			//for more explanations on degrees etc. check the drawings 
			

			float n;//distance between 2 consecutive points on a hair strand
			float gammaRad;//the angle(in radians) between previous point & +Z axis counterclockwise
			float alphaPrimeRad;//the angle(in radians) between newPoint & previous point @origin counterclockwise
			
			if (i == 0)//if first interpolatedPoint is inside head
			{
				n = length(hairPoint->interpolatedPoints[0] - hairPoint->startPoint);
				
				//first edge is minorLength --> distance between startPoint&minorCenter, is always |minorLength|
				//second edge is minorLength+0.1f
				//third edge is n
				alphaPrimeRad = acosf((minorLength*minorLength + (minorLength + 0.1f)*(minorLength + 0.1f) - n * n) /
					(2 * minorLength*(minorLength + 0.1f)));
				if (isnan(alphaPrimeRad))alphaPrimeRad = PI;

				gammaRad = atan2f(hairPoint->startPoint.y, hairPoint->startPoint.z);
			}
			else//non first interpolatedPoint is inside head
			{
				n = length(hairPoint->interpolatedPoints[i] - hairPoint->interpolatedPoints[i - 1]);
				
				//first edge is prev distance=length(previous point- minorcenter), which is sometimes equal to minorLength+0.1f
				//second edge is minorLength+0.1f
				//third edge is n
				float prevDistance = length(hairPoint->interpolatedPoints[i - 1] - minorCenter);

				alphaPrimeRad = acosf((prevDistance*prevDistance+ (minorLength + 0.1f)*(minorLength + 0.1f) - n * n) /
					(2 * prevDistance * (minorLength + 0.1f)));
				if (isnan(alphaPrimeRad))  alphaPrimeRad = PI;

				gammaRad = atan2f(hairPoint->interpolatedPoints[i - 1].y, hairPoint->interpolatedPoints[i - 1].z);
			}

			float3 prevPoint = (i > 0 ? hairPoint->interpolatedPoints[i - 1] : hairPoint->startPoint);//the point that comes before this point
			float3 newPoint;
			newPoint.x = hairPoint->interpolatedPoints[i].x;
			if ((prevPoint.y-n)*(prevPoint.y-n)+prevPoint.z*prevPoint.z>(minorLength+0.1f)*(minorLength+0.1f))//the hair falls down
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
			moveVectors = (float3*)malloc(sizeof(float3)*(hairPoint->interpolatedPointSize - i));
			for (int j = i+1; j < hairPoint->interpolatedPointSize; j++)
			{
				moveVectors[j-i-1] = hairPoint->interpolatedPoints[j] - hairPoint->interpolatedPoints[j-1];
			}
			moveVectors[hairPoint->interpolatedPointSize - i - 1] = hairPoint->endPoint - hairPoint->interpolatedPoints[hairPoint->interpolatedPointSize - 1];//set vector between end point - last interpolated point
			
			//set this point as new point
			hairPoint->interpolatedPoints[i] = newPoint;
			//add respective moveVectors to rest of the interpolatedPoints
			for (int j = i + 1; j < hairPoint->interpolatedPointSize; j++) hairPoint->interpolatedPoints[j] = hairPoint->interpolatedPoints[j - 1] + moveVectors[j - i - 1];

			//add moveVectors[-1] to endPoint
			hairPoint->endPoint = hairPoint->interpolatedPoints[hairPoint->interpolatedPointSize - 1] + moveVectors[hairPoint->interpolatedPointSize - i - 1];

			free(moveVectors);
		}
	}
	//check the endPoint
	if (length(hairPoint->endPoint-minorCenter) < minorLength+ 0.1f)
	{
		
		float n;//distance between 2 consecutive points on a hair strand
		float gammaRad;//the angle(in radians) between previous point & +Z axis counterclockwise
		float alphaPrimeRad;//the angle(in radians) between newPoint & +Z axis counterclockwise

		//calculate gammaRad & alphaPrimeRad
		if (hairPoint->interpolatedPointSize > 0)//when there is at least 1 interpolatedPoint
		{
			n = length(hairPoint->endPoint - hairPoint->interpolatedPoints[hairPoint->interpolatedPointSize - 1]);

			gammaRad = atan2f(hairPoint->interpolatedPoints[hairPoint->interpolatedPointSize - 1].y, hairPoint->interpolatedPoints[hairPoint->interpolatedPointSize - 1].z);

			//first edge is prev distance=length(previous point- minorcenter), which is sometimes equal to minorLength+0.1f
			//second edge is minorLength+0.1f
			//third edge is n
			float prevDistance = length(hairPoint->interpolatedPoints[hairPoint->interpolatedPointSize - 1] - minorCenter);

			alphaPrimeRad = acosf((prevDistance*prevDistance + (minorLength + 0.1f)*(minorLength + 0.1f) - n * n) /
				(2 * prevDistance * (minorLength + 0.1f)));
			if (isnan(alphaPrimeRad)) alphaPrimeRad = PI;//acosf can return nan
		}
		else//no interpolation --> previous point is starting point
		{
			n = length(hairPoint->endPoint - hairPoint->startPoint);
			gammaRad = atan2f(hairPoint->startPoint.y, hairPoint->startPoint.z);
			//first edge is minorlength
			//second edge is minorLength+0.1f
			//third edge is n
			alphaPrimeRad = acosf((2 * minorLength*(minorLength + 0.1f) - n * n) /
				(2 * minorLength*(minorLength + 0.1f)));
			if (isnan(alphaPrimeRad)) alphaPrimeRad = PI;//acosf can return nan
		}

		float3 prevPoint = (hairPoint->interpolatedPointSize > 0 ? hairPoint->interpolatedPoints[hairPoint->interpolatedPointSize - 1] : hairPoint->startPoint);//the point that comes before end point
		//rotate endPoint out of the head
		if ((prevPoint.y - n)*(prevPoint.y - n) + prevPoint.z*prevPoint.z>(minorLength + 0.1f)*(minorLength + 0.1f))//hair falls down
		{
			hairPoint->endPoint.y = prevPoint.y - n;
			hairPoint->endPoint.z = prevPoint.z;
		}
		else//hair follows curvature of the head
		{
			hairPoint->endPoint.y = (minorLength + 0.1f)*(sinf(alphaPrimeRad + gammaRad));
			hairPoint->endPoint.z = (minorLength + 0.1f)*(cosf(alphaPrimeRad + gammaRad));
		}
	}
}


//gets pointer to A hair strand, and index, option to print nothing
//DOES NOT USES INDEX TO ACCESS HAIRSTRAND
//uses index to print which hairStrand it is
//prints information about A hairStrand when dontPrint=false
//returns error count
int JourneyOfAHairStrand(hair *hairPoint, int index,bool dontPrint)
{
	//if (!dontPrint)std::cout <<std::endl<< "##############JOURNEY OF " << index << "th HAIR STRAND##############" << std::endl;
	//if (!dontPrint)std::cout << "head radius->(0 if you did not save original settings first)->" << oHeadRadius << std::endl;
	//if (!dontPrint)std::cout << "interpolated point size->" << hairPoint->interpolatedPointSize << std::endl;

	int errorCount=0;

	// ############################ hair length check ############################ 
	float hairLength=0;
	if (hairPoint->interpolatedPointSize == 0)
	{
		hairLength = length(hairPoint->endPoint - hairPoint->startPoint);
		if (!dontPrint)std::cout<<"length between all points on hair is same and equals to->"<< hairLength<< std::endl;
	}
	else//hairPoint->interpolatedPointSize > 1
	{
		float hairLengthPartition = 0,hairLengthPartition2=0;
		hairLengthPartition = length(hairPoint->interpolatedPoints[0] - hairPoint->startPoint);
		hairLength += hairLengthPartition;
		for (int i = 0; i < hairPoint->interpolatedPointSize - 1; i++)
		{
			hairLengthPartition2 = length(hairPoint->interpolatedPoints[i] - hairPoint->interpolatedPoints[i + 1]);
			if (roundf(hairLengthPartition - hairLengthPartition2)!=0)
			{
				if (!dontPrint) std::cout << "hair is not partitioned approximately equal to first partition between " << i << "th & " << i + 1 << "th interpolatedPoints" <<
					" /// " << hairLengthPartition << " - " << hairLengthPartition2 << " = " << hairLengthPartition - hairLengthPartition2 << std::endl;
				errorCount++;
			}
			hairLength += hairLengthPartition2;
		}
		hairLengthPartition2 = length(hairPoint->endPoint - hairPoint->interpolatedPoints[hairPoint->interpolatedPointSize-1]);
		if (roundf(hairLengthPartition - hairLengthPartition2) != 0)
		{
			if (!dontPrint)std::cout << "hair is not partitioned approximately equal to first partition between endPoint & last interpolatedPoint" <<
				" /// " << hairLengthPartition << " - " << hairLengthPartition2 << " = " << hairLengthPartition - hairLengthPartition2 << std::endl;
			errorCount++;
		}
		hairLength += hairLengthPartition2;
	}

	//if (!dontPrint)std::cout << "total hair length->" << hairLength << std::endl;

	// ############################ collision check ############################ 
	if (roundf( length(hairPoint->startPoint) - oHeadRadius )< 0)
	{
		if (!dontPrint)std::cout << "startPoint is inside the head" << std::endl;
		if (!dontPrint)std::cout << "startingPoint coordinates-> " << hairPoint->startPoint.x << "    " << hairPoint->startPoint.y
			<< "    " << hairPoint->startPoint.z << std::endl;
		errorCount++;
	}
	for (int i = 0; i < hairPoint->interpolatedPointSize; i++)
	{
		if (roundf(length(hairPoint->interpolatedPoints[i]) - oHeadRadius) < 0)
		{
			if (!dontPrint)std::cout << i<<"th interpolatedPoint is inside the head" << std::endl;
			if (!dontPrint)std::cout <<i<< "th interpolatedPoint coordinates-> " << hairPoint->interpolatedPoints[i].x << "    " << hairPoint->interpolatedPoints[i].y
				<< "    " << hairPoint->interpolatedPoints[i].z << std::endl;
			errorCount++;
		}
	}
	if (roundf(length(hairPoint->endPoint) - oHeadRadius )< 0)
	{
		if (!dontPrint)std::cout << "endPoint is inside the head" << std::endl;
		if (!dontPrint)std::cout << "endPoint coordinates-> " << hairPoint->endPoint.x << "    " << hairPoint->endPoint.y
			<< "    " << hairPoint->endPoint.z << std::endl;
		errorCount++;
	}

	//if (!dontPrint)std::cout << "ERROR COUNT: " << errorCount<<std::endl;
	return errorCount;

	//testing
	/*std::cout << "gammaRad of hair is->" << atan2f(hairPoint->startPoint.y, hairPoint->startPoint.z) << std::endl;
	std::cout << "length of the startPoint is->" << length(hairPoint->startPoint) <<"###### arctan->"<<
		atan2f(hairPoint->startPoint.y, hairPoint->startPoint.z) << std::endl;
	for (int i = 0; i < hairPoint->interpolatedPointSize; i++)
	{
		std::cout << "length of the interpolatedPoints[" << i << "] is->" << length(hairPoint->interpolatedPoints[i]) <<
			"###### arctan->" << atan2f(hairPoint->interpolatedPoints[i].y, hairPoint->interpolatedPoints[i].z) << std::endl;
	}
	std::cout << "length of the endPoint is->" << length(hairPoint->endPoint) <<
		"###### arctan->" << atan2f(hairPoint->endPoint.y, hairPoint->endPoint.z) << std::endl;
	std::cout << "startingPoint coordinates-> " << hairPoint->startPoint.x << "    " << hairPoint->startPoint.y
		<< "    " << hairPoint->startPoint.z << std::endl;
	for (int i = 0; i < hairPoint->interpolatedPointSize; i++)
	{
		std::cout << i << "th interpolatedPoint coordinates-> " << hairPoint->interpolatedPoints[i].x << "    " << hairPoint->interpolatedPoints[i].y
			<< "    " << hairPoint->interpolatedPoints[i].z << std::endl;
	}
	std::cout << "endPoint coordinates-> " << hairPoint->endPoint.x << "    " << hairPoint->endPoint.y
		<< "    " << hairPoint->endPoint.z << std::endl;*/


}

float hairLengthFinder(hair *hairPoint)
{
	float total = 0;
	float dif = 0;
	if (hairPoint->interpolatedPointSize > 0)
	{
		dif= length(hairPoint->startPoint - hairPoint->interpolatedPoints[0]);
		total += dif;
		for (int i = 0; i < hairPoint->interpolatedPointSize-1; i++)
		{
			dif = length(hairPoint->interpolatedPoints[i] - hairPoint->interpolatedPoints[i + 1]);
			total += dif;
		}
		dif= length(hairPoint->interpolatedPoints[hairPoint->interpolatedPointSize - 1] - hairPoint->endPoint);
		total += dif;
	}
	else total += length(hairPoint->endPoint - hairPoint->startPoint);
	//std::cout << "hairLength->" << total << std::endl;
	return total;
}

//frees hairPoints & oHairPoints
void FreeAllCPU(hair *hairPoints)
{
	for (int i = 0; i < oHairCount; i++)
	{
		free(hairPoints[i].interpolatedPoints);
		free(oHairPoints[i].interpolatedPoints);
	}
	free(hairPoints);
	free(oHairPoints);
}

/*
	y
	^
	|
	|------->z
	x is positive through inside
	(x,y,z)

	*eyes always look in +Z direction
	so the only place where hair does not come out is between (-x,-y,+z) & (+x,-y,+z)
	--this can be highly costumized

	in this case hair can come out from everywhere else other than(_,-y,+z)

*/


/* 
	2D rotating of a point around origin counterclockwise:
	x' = x cos f - y sin f
	y' = y cos f + x sin f
	from : https://www.siggraph.org/education/materials/HyperGraph/modeling/mod_tran/2drota.htm
*/
