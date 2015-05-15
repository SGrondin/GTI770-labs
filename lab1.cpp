/**
* GTI770 - Systemes intelligents et apprentissage machine
* Alessandro L. Koerich
* alessandro.koerich@etsmtl.ca
* 2015
*
* EXEMPLE 1: Feature extraction from RGB images
*                       Simpsons Family
**/

// INCLUDES
#include <cv.h> 			//OpenCV lib
#include <cvaux.h>			//OpenCV lib
#include <highgui.h>			//OpenCV lib
#include <stdio.h>
#include <math.h>
#include <stack>	

// DEFINES
#define NUM_SAMPLES 100
#define NUM_FEATURES 6
#define NUMBER_OF_CLASS 5

// Bart Train: 80 items: bart1.bmp - bart80.bmp
// Homer Train 62 items: homer1.bmp - homer62.bmp
// Bart Valid: 54 items: bart116.bmp - bart169.bmp
// Homer Valid: 37 items: homer88.bmp - homer124.bmp

float removeRecursive(int map[], int w, int h, int x, int y)
{
	std::stack<int> s;

	float amount = 0;
	s.push(x);
	s.push(y);

	while (!s.empty()) {
		y = s.top(); s.pop();
		x = s.top(); s.pop();
		map[x + y*w] = 0;
		amount++;

		if (x > 0 && map[(x-1) + y*w])     { s.push(x - 1); s.push(y); }
		if (x < w - 1 && map[(x+1) + y*w]) { s.push(x + 1); s.push(y); }
		if (y > 0 && map[x + (y-1)*w])     { s.push(x); s.push(y - 1); }
		if (y < h - 1 && map[x + (y+1)*w]) { s.push(x); s.push(y + 1); }
	}

	return amount;
}

int getNumberOfArea(int map[], int w, int h, float min, float max)
{
	int count = 0;
	float size = 0;

	for (int x=0; x<w; x++) {
		for (int y=0; y<h; y++) {
			if (map[x + y*w]) {	
				size = removeRecursive(map, w, h, x, y);
 
				//printf("%f\n", size);

				if (size > min && size < max)
				{
					count++;
				}
			}
		}
	}
	return count;
}

// MAIN
int main( int argc, char** argv )
{

	// Variable store pressed key
	int tecla;

	// General variables (loop)
	int h;
	int w;
	int ii;
	int jj;
	int iNum;

	// Variables to store the RGB values of a pixel
	unsigned char red;
	unsigned char blue;
	unsigned char green;

	// Feature vector [rows] [columns]
	// In fact it is a "matrix of features"
	float fVector[ NUM_SAMPLES ][ NUM_FEATURES ];

	// Variable filename
	static char cFileName[ 50 ] = {'\0'};
	FILE *fp;

	// Open a text file to store the feature vectors
	fp = fopen ("apprentissage-HBLOS.txt","w");

	if(fp == NULL) {
		perror("failed to open apprentissage-HBLOS.txt");
		return EXIT_FAILURE;
	}

	// OpenCV variables related to the image structure.
	// IplImage structure contains several information of the image (See OpenCV manual).
	IplImage *img 			= NULL;
	IplImage *processed 	= NULL;
	IplImage *threshold 	= NULL;

	// OpenCV variable that stores the image width and height
	CvSize tam;

	// OpenCV variable that stores a pixel value
	CvScalar element;

	// Fill fVector with zeros
	for ( ii = 0 ; ii < NUM_SAMPLES ; ii++ )
	{
		for ( jj = 0; jj < NUM_FEATURES; jj++ )
		{
			fVector[ii][jj] = 0.0;
		}
	}

	// Fill cFileName with zeros
	for ( ii = 0 ; ii < 50 ; ii++ )
	{
		cFileName[ ii ] = '\0';
	}


	// *****************************************************************************************************************************************
	// TRAINING SAMPLES
	// HOMER
	// Homer Train 62 items: homer1.bmp - homer62.bmp
	// *****************************************************************************************************************************************

	char path[NUMBER_OF_CLASS][80] = { {"Train/homer%03d.bmp"} , {"Train/bart%03d.bmp"}, {"Train/lisa%03d.bmp"}, {"Train/other%03d.bmp"}, {"Train/school%03d.bmp"} };
	char name[NUMBER_OF_CLASS][80] = { {"Homer"}, {"Bart"}, {"Lisa"}, {"Other"}, {"School"} };

	// Take all the image files at the range

	float numberPixelBeardHomer;
	float numberPixelBlueHomer;
	float numberPixelOrangeBart;
	float numberPixelBlueBart;
	float numberPixelRedLisa;
	float numberWhiteAreaLisa;
	bool isNewWhiteArea;

	for (ii =0; ii<NUMBER_OF_CLASS; ii++)
	{
		for ( iNum = 1; iNum <= 80; iNum++ )
		{


			// Build the image filename and path to read from disk
			sprintf ( cFileName, path[ii], (int)(iNum) );
			printf ( " %s\n", cFileName);

			// Load the image from disk to the structure img.
			// 1  - Load a 3-channel image (color)
			// 0  - Load a 1-channel image (gray level)
			// -1 - Load the image as it is  (depends on the file)

			img = cvLoadImage( cFileName, -1 );

			// Gets the image size (width, height) 'img'
			if (img == NULL) {
				continue;
			}

			// Gets the image size (width, height) 'img'
			tam = cvGetSize( img );

			// Creates a header and allocates memory (tam) to store a copy of the original image.
			// 1 - gray level image
			// 3 - color image
			// processed = cvCreateImage( tam, IPL_DEPTH_8U, 3);

			// Make a image clone and store it at processed and threshold
			processed 	= cvCloneImage( img );
			threshold  	= cvCloneImage( img );

			// Initialize variables with zero
			numberPixelBeardHomer = 0.0;
			numberPixelBlueHomer = 0.0;
			numberPixelOrangeBart = 0.0;
			numberPixelBlueBart = 0.0;
			numberPixelRedLisa = 0.0;
			int mapWhiteArea[img->width * img->height];

			// Loop that reads each image pixel
			for( h = 0; h < img->height; h++ ) // rows
			{
				for( w = 0; w < img->width; w++ ) // columns
				{
					// Read each channel and writes it into the blue, green and red variables. Notice that OpenCV considers BGR
					blue  	= ( (uchar *)(img->imageData + h*img->widthStep) )[ w*img->nChannels + 0 ];
					green 	= ( (uchar *)(img->imageData + h*img->widthStep) )[ w*img->nChannels + 1 ];
					red   	= ( (uchar *)(img->imageData + h*img->widthStep) )[ w*img->nChannels + 2 ];

					// Shows the pixel value at the screenl
					//printf( "pixel[%d][%d]= %d %d %d \n", h, w, (int)blue, (int)green, (int)red );

					// Here starts the feature extraction....

					// Detect and count the number of orange pixels
					// Verify if the pixels have a given value ( Orange, defined as R[240-255], G[85-105], B[11-22] ). If so, count it...
					if ( blue>=11 && blue<=22 && green>=85 && green<=105 &&  red>=240 && red<=255 )
					{
						numberPixelOrangeBart++;
					}

					if ( blue>=122 && blue<=142 && green>=0 && green<=18 &&  red>=0 && red<=10 )
					{
						numberPixelBlueBart++;
					}

					if ( blue>=97 && blue<=117 && green>=163 && green<=183 &&  red>=179 && red<=199 )
					{
						numberPixelBeardHomer++;
					}

					if ( blue>=163 && blue<=183 && green>=97 && green<=117 &&  red>=0 && red<=10 )
					{
						numberPixelBlueHomer++;
					}

					if ( blue>=0 && blue<=10 && green>=0 && green<=10 &&  red>=245 && red<=256 )
					{
						numberPixelRedLisa++;
					}

					if ( blue>=230 && blue<=256 && green>=230 && green<=256 &&  red>=230 && red<=256 )
					{
						mapWhiteArea[w+h*img->width] = 1;
					}
					else
					{
						mapWhiteArea[w+h*img->width] = 0;
					}	
				}
			}

			numberWhiteAreaLisa = getNumberOfArea(mapWhiteArea, img->width, img->height, (img->width * img->height) / 1000, (img->width * img->height) / 200);

			// Lets make our counting somewhat independent on the image size...
			// Compute the percentage of pixels of a given colour.
			// Normalize the feature by the image size
			numberPixelOrangeBart 	= numberPixelOrangeBart / ( (int)img->height * (int)img->width );
			numberPixelBlueBart  	= numberPixelBlueBart  / ( (int)img->height * (int)img->width );
			numberPixelBeardHomer  	= numberPixelBeardHomer  / ( (int)img->height * (int)img->width );
			numberPixelBlueHomer  	= numberPixelBlueHomer  / ( (int)img->height * (int)img->width );
			numberPixelRedLisa      = numberPixelRedLisa / ( (int)img->height * (int)img->width );

			// Store the feature value in the columns of the feature (matrix) vector
			fVector[iNum][1] = numberPixelOrangeBart;
			fVector[iNum][2] = numberPixelBlueBart;
			fVector[iNum][3] = numberPixelBeardHomer;
			fVector[iNum][4] = numberPixelBlueHomer;
			fVector[iNum][5] = numberPixelRedLisa;
			fVector[iNum][6] = numberWhiteAreaLisa;

			// Here you can add more features to your feature vector by filling the other columns: fVector[iNum][3] = ???; fVector[iNum][4] = ???;

			// Shows the feature vector at the screen
			printf( "\n%d %f %f %f %f %f %f", iNum, fVector[iNum][1], fVector[iNum][2], fVector[iNum][3], fVector[iNum][4], fVector[iNum][5], fVector[iNum][6]);

			// And finally, store your features in a file
			fprintf( fp, "%f,", fVector[iNum][1]);
			fprintf( fp, "%f,", fVector[iNum][2]);
			fprintf( fp, "%f,", fVector[iNum][3]);
			fprintf( fp, "%f,", fVector[iNum][4]);
			fprintf( fp, "%f,", fVector[iNum][5]);
			fprintf( fp, "%f,", fVector[iNum][6]);

			// IMPORTANT
			// Do not forget the label....
			fprintf( fp, "%s\n", name[ii]);


			// Finally, give a look at the original image and the image with the pixels of interest in green
			// OpenCV create an output window
			//cvShowImage( "Original", img );
			//cvShowImage( "Processed", processed );

			// Wait until a key is pressed to continue...
			//tecla = cvWaitKey(0);
		}
	}

	cvReleaseImage(&img);
	cvDestroyWindow("Original");

	cvReleaseImage(&processed);
	cvDestroyWindow("Processed");

	fclose(fp);

	return 0;
}
