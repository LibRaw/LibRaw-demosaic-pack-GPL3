//exposure correction before interpolation
// use FFD interpolation (provisional) from Luis Sanz (complete by J.Desmis for OMP) ==> calculate luminosity
// inspired from work Guillermo Luijk and Manuel LLorens(Perfectraw)
//Jacques Desmis 31 december 2010
// I use OMP
  // This function uses parameters:
  //      exposure (lineal): 2^(-8..0..8): currently 0.5 +3
  //      preserve (log)   : 0..8 : currently 0.1 1
//OMP for Libraw


void CLASS exp_bef(float expos, float preser)
{
	double dt,dT2;
	clock_t t1, t2,t3,t4;
	float Yp, exposure2, K, EV;
	float LUT[65536];
	int i;
	int row,col;
	ushort (*img)[4];// to save configuration (provisional interpolation)
#ifdef DCRAW_VERBOSE	
	if (verbose) fprintf (stderr,_("exposure before interpolation...[J.Desmis]\n"));
#endif	
	img = (ushort (*)[4]) calloc (height*width, sizeof *img);// to save interpolation
	memcpy(img,image,height*width*sizeof *img);//save configuration
	// I use FDD interpolate from Luis Sanz , very fast and good, one can change for another
	border_interpolate(3);
	t1 = clock();
#if defined (LIBRAW_USE_OPENMP)	
#pragma omp parallel  
#endif
{
int indx,c,f[2],g[2],d;
int u=width,v=2*u,w=3*u;
#if defined (LIBRAW_USE_OPENMP)	
#pragma omp for schedule(dynamic) nowait
#endif
	for (int row=3; row < height-3; row++)
		for (int col=3+(FC(row,1)&1),c=FC(row,col),indx=row*width+col; col<u-3; col+=2,indx+=2)
			if ((abs(image[indx-2][c]-image[indx][c]-image[indx-1][1]+image[indx+1][1])+abs(image[indx][c]-image[indx+2][c]-image[indx-1][1]+image[indx+1][1])+abs(image[indx][c]-image[indx-2][c]-image[indx-1][1]+image[indx-3][1])+abs(image[indx][c]-image[indx+2][c]-image[indx+1][1]+image[indx+3][1]))<(abs(image[indx-v][c]-image[indx][c]-image[indx-u][1]+image[indx+u][1])+abs(image[indx][c]-image[indx+v][c]-image[indx-u][1]+image[indx+u][1])+abs(image[indx][c]-image[indx-v][c]-image[indx-u][1]+image[indx-w][1])+abs(image[indx][c]-image[indx+v][c]-image[indx+u][1]+image[indx+w][1])))
				image[indx][1]=ULIM((image[indx-1][1]+image[indx+1][1]+(2*image[indx][c]-image[indx-2][c]-image[indx+2][c])/2)>>1,image[indx-1][1],image[indx+1][1]);
			else
				image[indx][1]=ULIM((image[indx-u][1]+image[indx+u][1]+(2*image[indx][c]-image[indx-v][c]-image[indx+v][c])/2)>>1,image[indx-u][1],image[indx+u][1]);
#if defined (LIBRAW_USE_OPENMP)
#pragma omp for schedule(dynamic) nowait
#endif
	for (int row=1; row < height-1; row++)
		for (int col=1+(FC(row,2)&1),c=FC(row,col+1),d=2-c,indx=row*width+col; col<u-1; col+=2,indx+=2) {
			image[indx][c]=CLIP((image[indx-1][c]+image[indx+1][c]+2*image[indx][1]-image[indx-1][1]-image[indx+1][1])>>1);
			image[indx][d]=CLIP((image[indx-u][d]+image[indx+u][d]+2*image[indx][1]-image[indx-u][1]-image[indx+u][1])>>1);
		}
#if defined (LIBRAW_USE_OPENMP)
#pragma omp for schedule(dynamic) nowait
#endif
	for (int row=1; row < height-1; row++)
		for (int col=1+(FC(row,1)&1),c=2-FC(row,col),indx=row*width+col; col<u-1; col+=2,indx+=2) {
			f[0]=abs(image[indx-1-u][c]-image[indx+u+1][c])+abs(image[indx-1-u][1]-image[indx][1])+abs(image[indx+u+1][1]-image[indx][1]);
			f[1]=abs(image[indx+1-u][c]-image[indx+u-1][c])+abs(image[indx+1-u][1]-image[indx][1])+abs(image[indx+u-1][1]-image[indx][1]);
			g[0]=image[indx-1-u][c]+image[indx+u+1][c]+2*image[indx][1]-image[indx-1-u][1]-image[indx+u+1][1];
			g[1]=image[indx+1-u][c]+image[indx+u-1][c]+2*image[indx][1]-image[indx+1-u][1]-image[indx+u-1][1];
			image[indx][c]=CLIP(g[f[0]>=f[1]]>>1);
		}
		}
	t2 = clock();
// calculate CIE luminosity
float *YY;
YY = (float *)calloc(width*height,sizeof *YY);// for CIE luminosity

#pragma omp parallel default(shared)  
{
#if defined (LIBRAW_USE_OPENMP)	
#pragma omp for  
#endif
	 for(int i=0;i<height*width;i++){
           YY[i]=CLIP(0.299*(float)image[i][0]+0.587*(float)image[i][1]+0.114*(float)image[i][2]); // CIE luminosity
				}
}				
	memcpy(image,img,width*height*sizeof *img); // restore configuration
	free(img);	  //free memory
	//exposure correction inspired from G.Luijk
 if(preser==0.0){	// protect highlights 
#ifdef DCRAW_VERBOSE
  if(verbose)  fprintf (stderr,_("without highlight preservation\n"));
#endif  
#if defined (LIBRAW_USE_OPENMP)	  
#pragma omp parallel for  shared(expos)
#endif
	    for(int i=0;i<height*width;i++){
		for(int c=0;c<4;c++) image[i][c]=CLIP((float)image[i][c]*expos);}
  }else{
    // Exposure correction with highlight preservation
#ifdef DCRAW_VERBOSE	
  if(verbose)  fprintf (stderr,_("with highlight preservation\n"));
#endif  
    if(expos>1){
      K=65535/expos*exp(-preser*log((double) 2));
      for(int j=0;j<=65535;j++) LUT[(int)j]=CLIP(((65535-K*expos)/(65535-K)*(j-65535)+65535)/j);

#if defined (LIBRAW_USE_OPENMP)		  
#pragma omp parallel for  shared(expos)
#endif
      for(int i=0;i<height*width;i++){
        if(YY[i]<K){
          for(int c=0;c<4;c++) image[i][c]=CLIP((float)image[i][c]*expos); 
        }else{
          exposure2=LUT[(int)YY[i]];
          for(int c=0;c<4;c++)  image[i][c]=CLIP((float)image[i][c]*exposure2);
        }
      }
    }
	else{
      float EV=log(expos)/log(2.0);                              // Convert exp. linear to EV
      float K=65535.0*exp(-preser*log((double) 2));
      for(int j=0;j<=65535;j++) LUT[(int)j]=CLIP(exp(EV*(65535.0-j)/(65535.0-K)*log((double) 2)));
#if defined (LIBRAW_USE_OPENMP)		  
#pragma omp parallel for  shared(expos) 
#endif  
      for(int i=0;i<height*width;i++){
        if(YY[i]<K){
          for(int c=0;c<4;c++)  image[i][c]=CLIP((float)image[i][c]*expos);
        }else{
          float exposure2=LUT[(int)YY[i]];
          for(int c=0;c<4;c++)  image[i][c]=CLIP((float)image[i][c]*exposure2);
        }
      }
    }	
}
free(YY);

	t4=clock();	
		dT2 = ((double)(t4 - t1)) / CLOCKS_PER_SEC;
#ifdef DCRAW_VERBOSE		
		if (verbose) fprintf (stderr,_("\t exposure before interpolation done in %5.3fs.\n"),dT2);
#endif		
		
}