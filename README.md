# Historical-Precipitation-Analysis
The script herein this project attempts to generate gridded output from in situ observations of snowfall specifically. 
This script can be altered to include interpolations of different precipiation types (rain, sleet, ice, etc...) if observations are spacious enough...
Simply alter the colorbar accordingly.

The script can be broken down such that;
1) Observations are obtained from the GHCND server/archive.
1.a) Flagged observations are imediately removed using Pyspark.
1.b) Metadata (coordinates) are joined and processed.
2) Observations are processed in an spatial outlier algorithm which is a function of IDW and query search radius.
2.a) Given an observation (query point), the IDW mean is calculated from all observations within a search radius.
2.c) Given an observation (query point), the sample standard deviation is calculated from all observations within a search radius.
2.d) The query point is concidered an outlier if its Z-Score exceeds certain thresholds.
2.e) Continue to iterate the outlier algorithm until no additional outliers are found.
The outlier algorithm was altered depending upon the event.
3) Dumby points (z=0) are added to the irregular grid. Dumby points are added if any given point is a certain distance from an in situ observation.
4) I added an option to interpolate values for locations that are void of observers (Maine).
4.a) 0 == if there are observations north, east, south, and west of a query point.
     1 == interpolate regardless
     3 == do not interpolate
The interpolation is IDW.
5) Combine all dumby/in situ observations into an n-3 matrix (lon,lat,z)
6) Build a Kriging model (spherical/hole-effect performed best).
7) Interpolate the Kriging model onto a regular grid.
8) Plot its results.

Pending the number of observations and size of the analysis domain, this may crash your computer. I recommend using this via an HPC server.
