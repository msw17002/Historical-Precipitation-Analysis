#!/usr/bin/env python
# coding: utf-8

import os
from os import listdir
import matplotlib        as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import sin, cos, sqrt, atan2, atan, radians
from shapely.geometry import MultiLineString
import cartopy.io.shapereader as shpreader
from matplotlib.colors import BoundaryNorm
from cartopy.io.shapereader import Reader
from pykrige.ok import OrdinaryKriging
#from netCDF4 import Dataset
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
import pykrige.kriging_tools as kt
from itertools import compress
import cartopy.crs as ccrs
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import math
import os

# Download and create the states, land, and oceans using cartopy features
#states = cfeature.NaturalEarthFeature(category='cultural', scale='50m',
#                                      facecolor='none',
#                                      name='admin_1_states_provinces_shp')
#land = cfeature.NaturalEarthFeature(category='physical', name='land',
#                                    scale='50m',
#                                    facecolor=cfeature.COLORS['land'])
ocean = cfeature.NaturalEarthFeature(category='physical', name='ocean',
                                     scale='50m',
                                     facecolor=cfeature.COLORS['water'])

#####################################################################################################################
### SETTING CONSTANTS
#####################################################################################################################
# Plotting window
West_NE    = -82
East_NE    = -66
South_NE   = 37
North_NE   = 48
dxdy       = 0.2
Buffer     = 2
STD_HIGH   = 100
man_adj    = 0.1
RadEarth   = 6371000 #m Euclidean Dist.
ME_AUG     = 0      #0 interpo if boxxed, 1 interp
KType      = 'hole-effect' #'hole-effect' #'spherical'
# Shapefiles for cartopy
shp_path      = '/shared/stormcenter/AirMG/ISD_Evaluation/Static_Files/Shapefiles_Cartopy'
fname_usa     = shp_path+'/United_States_States_5m/cb_2016_us_state_5m.shp'
fname_usa_cnt = shp_path+'/cb_2018_us_county_5m/cb_2018_us_county_5m.shp'
fname_chm     = shp_path+'/Lake_Champlain/c0100e10-b780-4d5f-b0df-43d1a79a49ad2020410-1-seteko.xnxnj.shp'
fname_can     = shp_path+'/Canada_trsmd/canada_tr.shp'
fname_mask    = shp_path+'/Masks/Mask_NE.shp'
fname_ocean   = shp_path+'/ne_10m_ocean/ne_10m_ocean.shp'
fname_roads   = shp_path+'/ne_10m_roads/ne_10m_roads_python/ne_10m_roads_python.shp'

# Creating colorbar
clevs     = [0,0.1,1,2,3,4,6,8,12,18,24,30,36,48,60,72,96]
cmap_data = [(0.96456693, 0.96456693, 0.96456693),
                 (0.77952756, 0.86614173, 0.92125984),
                 (0.51574803, 0.73228346, 0.86220472),
                 (0.32283465, 0.58661417, 0.77952756),
                 (0.18897638, 0.42913386, 0.66535433),
                 (0.17716535, 0.28740157, 0.63385827),
                 (1.        , 1.        , 0.6496063 ),
                 (1.        , 0.80314961, 0.16535433),
                 (1.        , 0.6023622 , 0.16141732),
                 (0.88188976, 0.21259843, 0.15748031),
                 (0.68503937, 0.15354331, 0.16929134),
                 (0.50787402, 0.16535433, 0.16535433),
                 (0.33858268, 0.16535433, 0.16535433),
                 (0.83070866, 0.83070866, 1.        ),
                 (0.68110236, 0.62204724, 0.87007874),
                 (0.57086614, 0.43307087, 0.7007874 )]
cmap_snow      = mcolors.ListedColormap(cmap_data, 'precipitation')
norm_snow      = mcolors.BoundaryNorm(clevs, cmap_snow.N)
#####################################################################################################################
### DEFINING FUNCTIONS
#####################################################################################################################
# Colorbar function -------------------------------------------------------------------------------
def make_colorbar(ax, mappable, **kwargs):
    divider = make_axes_locatable(ax)
    orientation = kwargs.pop('orientation', 'vertical')
    if orientation == 'vertical':
        loc = 'right'
    elif orientation == 'horizontal':
        loc = 'bottom'
    cax = divider.append_axes(loc, '5%', pad='3%', axes_class=mpl.pyplot.Axes)
    ax.get_figure().colorbar(mappable, cax=cax, orientation=orientation,label="inch", ticks=[0,0.1,1,2,3,4,6,8,12,18,24,30,36,48,60,72,96])
# Spatial Outlier Algorithm -------------------------------------------------------------------------------
def Outlier_Check(XYZ,dz,pwr):
    dsize    = XYZ.shape
    n        = int((dsize[0]))
    print("All available observations pre-outlier analysis: " + str(n))
    #Pre-alo Variables
    IDW      = np.zeros((n,n))
    widw     = np.zeros((n,n))
    hij      = np.zeros((n,n))
    #Loop through each point
    for i in range(n):
        #Determine euclidean distance (km) from each observation [RadEarth*c/1000]
        LonD        = np.radians(XYZ[:,0])
        LatD        = np.radians(XYZ[:,1])
        delta_phi   = LatD[i]-LatD
        delta_lamda = LonD[i]-LonD
        a           = np.sin(delta_phi/2)**2 + np.cos(LatD[i])*np.cos(LatD)*np.sin(delta_lamda/2.0)**2
        a[a==0]     = np.nan
        c           = 2*np.arctan(np.sqrt(a)/np.sqrt(1-a)) #getting errors in a
        #Replace these values with nan... avoid error message
        c[c==0]     = np.nan
        #Determine Inverse Distance Weight between each observation
        IDW[i,:]    = 1/(RadEarth*c/1000)**pwr
        IDW[i,i]    = np.nan
        #Index observations for each stations
        widw[:,i]   = XYZ[i,2]
        widw[i,i]   = np.nan
        #Determine Euclidean distance b/n each station
        hij[i,:]    = RadEarth*c/1000
        hij[i,i]    = 99999
    #Are the observations w/n threshold that has been defined
    Eucl_Logical    = hij<dz #getting errors in hij
    Sum_row_Eucl    = np.nansum(IDW*Eucl_Logical, axis=1)
    #Weighted distance b/n observations as a ratio
    Wij = np.full((n, n),np.nan)
    for i in range(n):
        if Sum_row_Eucl[i] > 0:
            Wij[i,:] = np.round(IDW[i,:]/Sum_row_Eucl[i],2)        
        Wij[i,i] = np.nan
    #Observations with no cooresponding stations within a given threshold will have Wij == inf... 
    #Replace these values with nan
    std_idw           = np.std(XYZ[:,2])
    weighted_variable = np.nansum(Wij*widw*Eucl_Logical,axis=1)
    #Z-score for a normal distribution
    Zscore_i          = (XYZ[:,2]-weighted_variable)/std_idw
    Logical_array     = [np.logical_and(Zscore_i > -1.280, Zscore_i < STD_HIGH)]
    #non_outliers      = abs(Zscore_i)<1.960
    #Return outliers
    XYZ_outliers      = XYZ[Logical_array[0]==False]
    #Return nonoutliers
    XYZ_nonoutliers   = XYZ[Logical_array[0]==True]
    print("Outliers detected: " + str(len(XYZ_outliers)))
    print("Specifications: Interpolation == IDW to the " + str(pwr) + " power")
    print("Specifications: Search Radius == " + str(dz) + "km")
    print("----------------------------------------------")
    return(XYZ_nonoutliers,XYZ_outliers,Zscore_i)
# Adding '0' inch observations function -------------------------------------------------------------------------------
def Dumby_Check(lons, lats, query_lon, query_lat, dz):
    hij        = np.zeros((len(lons),1))
    for z in range(len(lons)):
        # Constant
        R = RadEarth
        # Observation points
        lat1 = lats[z]
        lon1 = lons[z]
        # Query/grid point
        lat2 = query_lat
        lon2 = query_lon
        # Calculation
        phi1, phi2 = math.radians(lat1), math.radians(lat2) 
        dphi       = math.radians(lat2 - lat1)
        dlambda    = math.radians(lon2 - lon1)
        a          = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        # Euclidean distance in km
        hij[z,0]   = ((2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a)))/1000)
    hij            = np.sum(hij<=dz)
    # Return the number of stations within dz kms of an observation.
    return(hij)
# Points for interpolation check -------------------------------------------------------------------------------
def Density_Check(lons,lats,data,dz,dp):
    XYZ      = np.concatenate((np.expand_dims(lons,1), np.expand_dims(lats,1), np.expand_dims(data,1)), axis=1)
    print("----------------------------------------------")
    print("Initial number of observations: " + str(len(XYZ)))
    n        = len(XYZ)
    #Pre-alo Variables
    hij      = np.zeros((n,n))
    #Loop through each point
    for i in range(n):
        #Determine euclidean distance (km) from each observation [RadEarth*c/1000]
        LonD        = np.radians(XYZ[:,0])
        LatD        = np.radians(XYZ[:,1])
        delta_phi   = LatD[i]-LatD
        delta_lamda = LonD[i]-LonD
        a           = np.sin(delta_phi/2)**2 + np.cos(LatD[i])*np.cos(LatD)*np.sin(delta_lamda/2.0)**2
        a[a==0]     = np.nan
        c           = 2*np.arctan(np.sqrt(a)/np.sqrt(1-a)) #getting errors in a
        #Replace these values with nan... avoid error message
        c[c==0]     = np.nan
        #Determine Euclidean distance b/n each station
        hij[i,:]    = RadEarth*c/1000
        hij[i,i]    = 99999
    #Are the observations w/n threshold that has been defined
    Eucl_Logical    = hij<dz #getting errors in hij
    Eucl_Logical    = np.expand_dims(np.sum(Eucl_Logical,axis=1),1)
    XYZ             = np.asarray(list(compress(XYZ,(Eucl_Logical>dp)==True)))
    print("Clustered number of observations: " + str(int(len(XYZ))))
    print("Specifications: Distance == " + str(dz) + "km Point Density == " + str(dp))
    print("----------------------------------------------")
    return(XYZ)
def Dumby_Check_02(XYZ, query_lon, query_lat, dz):
    hij        = np.zeros((len(XYZ),1))
    for z in range(len(XYZ)):
        # Constant
        R = RadEarth
        # Observation points
        lat1 = XYZ[z,1]
        lon1 = XYZ[z,0]
        # Query/grid point
        lat2 = query_lat
        lon2 = query_lon
        # Calculation
        phi1, phi2 = math.radians(lat1), math.radians(lat2) 
        dphi       = math.radians(lat2 - lat1)
        dlambda    = math.radians(lon2 - lon1)
        a          = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        # Euclidean distance in km
        hij[z,0]   = ((2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a)))/1000)
    XYZ = XYZ[hij[:,0]<dz]
    # Return the number of stations within dz kms of an observation.
    return(XYZ)

def simple_idw(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)

    # In IDW, weights are 1 / distance
    weights = 1.0 / dist

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi

def distance_matrix(x0, y0, x1, y1):
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    # Make a distance matrix between pairwise observations
    # Note: from <http://stackoverflow.com/questions/1871536>
    # (Yay for ufuncs!)
    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])

    return np.hypot(d0, d1)
#####################################################################################################################
### LOAD AND PRE-PROCESS DATA, AND INTERPOLATE THE DATA
#####################################################################################################################
Path_Files     = ''
File_csv       = pd.read_csv('GHCND_List.csv')

for zzz in range(1):#len(File_csv)):
#for zzz in range(59,len(File_csv)):
    print(zzz)
    # Load and pre-process
    #print("./complete_list/complete_list_20141209_20141214_SNOW.csv")
    #KType        = 'hole-effect' #'hole-effect' #'spherical'
    Kriging_Grid = 0.025
    Control_Grid = dxdy
    #Testing    = pd.read_csv("./complete_list_20201029_20201030_SNOW.csv")
    #File       = "./complete_list_20201029_20201030_SNOW.csv"
    Testing    = pd.read_csv(File_csv['List'][zzz])
    File       = File_csv['List'][zzz]
    print(File)
    File       = File.replace('../complete_list/complete_list_', '')
    #File       = File.replace('./complete_list_', '')
    File_title = File.replace('_SNOW.csv', '')
    File_txt   = File.replace('_SNOW.csv', '.txt')
    File_png   = File.replace('_SNOW.csv', '.png')
    File_pdf   = File.replace('_SNOW.csv', '.pdf')
    title_str  = File_title[0:4]+"-"+File_title[4:6]+"-"+File_title[6:8]+" to "+File_title[9:13]+"-"+File_title[13:15]+"-"+File_title[15:17]

    Testing    = Testing.astype(float)
    Testing    = Testing[Testing['0']>-83] #87
    lons       = Testing['0']#.astype(float)
    lats       = Testing['1']#.astype(float)
    data       = Testing['2']#.astype(float)
    XYZ        = np.concatenate((np.expand_dims(np.asarray(lons),1), np.expand_dims(np.asarray(lats),1), 
                                         np.expand_dims(np.asarray(data),1)),axis=1)
    print("Number of Observations Pre-Outliers: "+str(len(XYZ)))
    #XYZ,XYZ_outliers,Zscore_i = Outlier_Check(XYZ,50,2) 
    XYZ_obs        = []
    XYZ_out        = []
    do             = len(XYZ)
    doutlier       = [np.nan,np.nan] #start
    i              = 0
    while (doutlier[-1]-doutlier[-2])!= 0:
        i                         = i+1
        XYZ,XYZ_outliers,Zscore_i = Outlier_Check(XYZ,50,2)
        print(len(XYZ))
        doutlier.append(do-len(XYZ))
        XYZ_obs.append(XYZ)
        XYZ_out.append(XYZ_outliers)
        print("Snow Iteration: "+str(i)+" Total Outliers: "+str(doutlier[-1]))
        if i>10: break
    XYZ_obs = np.unique(np.vstack(XYZ_obs),axis=0)
    XYZ_out = np.unique(np.vstack(XYZ_out),axis=0)
    print("Number of Observations Post-Outliers: "+str(len(XYZ_obs)))
    print("Number of Outliers Post-Outliers: "+str(len(XYZ_out)))
    #XYZ_nonoutliers = Passed_ite
    #print("Num. of outliers <-1.280 = "+str(sum(Zscore_i<-1.280)))
    #print("Num. of outliers <"+str(STD_HIGH)+" = "+str(sum(Zscore_i>STD_HIGH)))
    # Begin to set grid for Kriging interpolation
    grid_space     = Control_Grid
    grid_lon       = np.arange(West_NE , East_NE , grid_space) #grid_space is the desired delta/step of the output array 
    grid_lat       = np.arange(South_NE, North_NE, grid_space)
    xintrp, yintrp = np.meshgrid(grid_lon, grid_lat)
    # Add dumby points [obs == 0 inch]
    dumby_lon      = []
    dumby_lat      = []
    n_iterations   = xintrp.shape
    for x in range(n_iterations[0]):
        for y in range(n_iterations[1]):
            query_log = Dumby_Check(XYZ[:,0], XYZ[:,1], xintrp[x,y], yintrp[x,y], 50)
            if query_log==0: 
                dumby_lon.append(xintrp[x,y])
                dumby_lat.append(yintrp[x,y])
    dumby_data                             = np.concatenate((np.expand_dims(np.asarray(dumby_lon),1), 
                                                             np.expand_dims(np.asarray(dumby_lat),1), 
                                                             np.zeros((len(dumby_lat),1))),axis=1)
    N_Cpts                                 = len(dumby_data)
    XYZ_nonoutliers                        = np.concatenate((dumby_data,XYZ),axis=0)
    if ME_AUG==0:
        for i in range(len(XYZ_nonoutliers)):
            if XYZ_nonoutliers[i,2]==0:
                query_lon = XYZ_nonoutliers[i,0]
                query_lat = XYZ_nonoutliers[i,1]
                if (query_lon<-66) & (query_lon>-72) & (query_lat<48) & (query_lat>44):
                    lon_idx   = query_lon - XYZ_nonoutliers[:,0]
                    lat_idx   = query_lat - XYZ_nonoutliers[:,1]
                    quad_tl   = XYZ_nonoutliers[(lon_idx<0) & (lat_idx>0)]   #- +
                    quad_tr   = XYZ_nonoutliers[(lon_idx>0) & (lat_idx>0)]   #+ +
                    quad_bl   = XYZ_nonoutliers[(lon_idx>0) & (lat_idx<0)]   #+ -
                    quad_br   = XYZ_nonoutliers[(lon_idx<0) & (lat_idx<0)]   #- -
                    if (len(quad_tl[:,2]!=0)>0) & (len(quad_tr[:,2]!=0)>0) & (len(quad_bl[:,2]!=0)>0) & (len(quad_br[:,2]!=0)>0):
                        quad_tl   = Dumby_Check_02(quad_tl, query_lon, query_lat, 200)
                        quad_tl   = quad_tl[quad_tl[:,2]!=0]
                        quad_tr   = Dumby_Check_02(quad_tr, query_lon, query_lat, 200)
                        quad_tr   = quad_tr[quad_tr[:,2]!=0]
                        quad_bl   = Dumby_Check_02(quad_bl, query_lon, query_lat, 200)
                        quad_bl   = quad_bl[quad_bl[:,2]!=0]
                        quad_br   = Dumby_Check_02(quad_br, query_lon, query_lat, 200)
                        quad_br   = quad_br[quad_br[:,2]!=0]
                        if (len(quad_tl[:,2]!=0)>0) & (len(quad_tr[:,2]!=0)>0) & (len(quad_bl[:,2]!=0)>0) & (len(quad_br[:,2]!=0)>0):
                            quad_int_m           = np.concatenate((quad_tl,quad_tr,quad_bl,quad_br),axis=0)
                            quad_int_q           = simple_idw(quad_int_m[:,0], quad_int_m[:,1], quad_int_m[:,2], query_lon, query_lat)
                            XYZ_nonoutliers[i,2] = quad_int_q[0]
                            #print("INT = "+str(quad_int_q[0])+" "+str(i))
                            #print(str(query_lat)+","+str(query_lon)+"")
    elif ME_AUG==1: 
        for i in range(len(XYZ_nonoutliers)):
            if XYZ_nonoutliers[i,2]==0:
                query_lon = XYZ_nonoutliers[i,0]
                query_lat = XYZ_nonoutliers[i,1]
                if (query_lon<-66) & (query_lon>-72) & (query_lat<48) & (query_lat>44):
                    lon_idx   = query_lon - XYZ_nonoutliers[:,0]
                    lat_idx   = query_lat - XYZ_nonoutliers[:,1]
                    quad_tl   = XYZ_nonoutliers[(lon_idx<0) & (lat_idx>0)]   #- +
                    quad_tr   = XYZ_nonoutliers[(lon_idx>0) & (lat_idx>0)]   #+ +
                    quad_bl   = XYZ_nonoutliers[(lon_idx>0) & (lat_idx<0)]   #+ -
                    quad_br   = XYZ_nonoutliers[(lon_idx<0) & (lat_idx<0)]   #- -
                    quad_tl   = Dumby_Check_02(quad_tl, query_lon, query_lat, 20)
                    quad_tl   = quad_tl[quad_tl[:,2]!=0]
                    quad_tr   = Dumby_Check_02(quad_tr, query_lon, query_lat, 20)
                    quad_tr   = quad_tr[quad_tr[:,2]!=0]
                    quad_bl   = Dumby_Check_02(quad_bl, query_lon, query_lat, 20)
                    quad_bl   = quad_bl[quad_bl[:,2]!=0]
                    quad_br   = Dumby_Check_02(quad_br, query_lon, query_lat, 20)
                    quad_br   = quad_br[quad_br[:,2]!=0]
                    if (len(quad_tl[:,2]!=0)+len(quad_tr[:,2]!=0)+len(quad_bl[:,2]!=0)+len(quad_br[:,2]!=0))>0:
                        quad_int_m           = np.concatenate((quad_tl,quad_tr,quad_bl,quad_br),axis=0)
                        quad_int_q           = simple_idw(quad_int_m[:,0], quad_int_m[:,1], quad_int_m[:,2], query_lon, query_lat)
                        XYZ_nonoutliers[i,2] = quad_int_q[0]
                            #print("INT = "+str(quad_int_q[0])+" "+str(i))
                            #print(str(query_lat)+","+str(query_lon)+"")
    # Nonoutlier data including 0 dumby points
    Lon   = XYZ_nonoutliers[:,0].astype(float)
    Lat   = XYZ_nonoutliers[:,1].astype(float)
    data  = XYZ_nonoutliers[:,2].astype(float)
    
    XYZ_nonoutliers = XYZ_nonoutliers[XYZ_nonoutliers[:,2]!=0]
    Region_Log      = np.empty([len(XYZ_nonoutliers),1], dtype=object)
    for i in range(len(XYZ_nonoutliers)):
        if (XYZ_nonoutliers[i,1] < North_NE) and (XYZ_nonoutliers[i,1] > South_NE) and \
           (XYZ_nonoutliers[i,0] <  East_NE) and (XYZ_nonoutliers[i,0] >  West_NE): Region_Log[i,0] = '1'
        else: Region_Log[i,0] = '0'
    XYZ_nonoutliers = XYZ_nonoutliers[Region_Log[:,0]=='1']
    #####################################################################################################################
    ### SPHERICAL KRIGIN INTERPOLATION ONTO GRID
    #####################################################################################################################
    print('Ord. Kriging #1')
    grid_lon   = np.arange(West_NE , East_NE , Kriging_Grid)
    grid_lat   = np.arange(South_NE, North_NE, Kriging_Grid)
    North      = np.amax(Lat)+Buffer
    East       = np.amax(Lon)-Buffer
    South      = np.amin(Lat)+Buffer
    West       = np.amin(Lon)-Buffer
    print('Ord. Kriging #2')
    #OK      = OrdinaryKriging(Lon, Lat, data, variogram_model='spherical', verbose=True,
    #                          enable_plotting=False, coordinates_type='geographic', enable_statistics=True)
    OK      = OrdinaryKriging(Lon, Lat, data, variogram_model=KType, verbose=True,
                                  enable_plotting=False, enable_statistics=False, coordinates_type='geographic')
    print('Ord. Kriging #3')
    grid_lon   = np.arange(West_NE , East_NE , Kriging_Grid)
    grid_lat   = np.arange(South_NE, North_NE, Kriging_Grid)
    z1, ss1 = OK.execute('grid', grid_lon, grid_lat)
    z1[z1<man_adj] = 0 #change negative values to 0
    #coords=np.array((grid_lon.ravel(), grid_lat.ravel(), z1.ravel()).T
    #coords=pd.DataFrame(coords)
    #coords.rename(columns={0:'Lon',1:'Lat',2:'Snow_inch'}, inplace=True)
    #coords.to_csv("Grid"+"_"+File_txt, sep=',',index=True)
    print('Ord. Kriging #4')
    xintrp, yintrp = np.meshgrid(grid_lon, grid_lat)
    print('Ord. Kriging #5')
    #####################################################################################################################
    ### PLOTTING
    #####################################################################################################################
    Lon   = XYZ_nonoutliers[:,0].astype(float)
    Lat   = XYZ_nonoutliers[:,1].astype(float)
    data  = XYZ_nonoutliers[:,2].astype(float)
    # Figure
    fig = plt.figure(figsize=(10, 10))
    #mapping resources
    ax  = plt.subplot(1,1,1, projection=ccrs.Mercator())
    ax.set_extent([West_NE, East_NE, South_NE, North_NE],crs=ccrs.PlateCarree())
    ax.set_title('Historical Snowfall Interpolation: '+'\n'+title_str, fontsize=24, fontweight='bold')
    #ax.add_geometries(Reader(fname_usa_cnt).geometries(), ccrs.PlateCarree(), facecolor="none", edgecolor='k', lw=0.25)
    cb = plt.contourf(xintrp, yintrp, z1, np.asarray(clevs), cmap=cmap_snow, norm=norm_snow, vmin = 0, vmax = 96, alpha=0.90, transform=ccrs.PlateCarree())
    
    #contours 
    CS = ax.contour(xintrp, yintrp, z1,  [12,24,36,48],linestyles='dashed', colors='k',transform=ccrs.PlateCarree())
    ax.clabel(CS, fontsize=16,inline=True, fmt='%1.0f')
    
    #mapping resources
    ax.add_feature(ocean)
    ax.add_geometries(Reader(fname_can).geometries(), ccrs.PlateCarree(), facecolor="w", hatch='\\\\\\\\',  edgecolor='black', lw=0.7)
    
    #observations
    cb = plt.scatter(Lon[data!=0], Lat[data!=0], c=data[data!=0], s=5, cmap=cmap_snow, norm=norm_snow, alpha=1, edgecolors='k', linewidths=0.25,
                     marker='o', transform=ccrs.PlateCarree())

    #mapping resources
    ax.add_geometries(Reader(fname_usa_cnt).geometries(), ccrs.PlateCarree(), facecolor="none", edgecolor='silver', lw=0.25, alpha=0.40)
    ax.add_feature(cfeature.LAKES, facecolor="lightcyan", edgecolor='black', lw=0.5)
    ax.add_geometries(Reader(fname_usa).geometries(), ccrs.PlateCarree(), facecolor="none", edgecolor='k', lw=1)

    #colorbar
    make_colorbar(ax, cb, pad=0)

    #graticule
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.75, color='gray', alpha=0.8, linestyle='--')
    gl.ylocator = mticker.FixedLocator([32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54])
    gl.xlocator = mticker.FixedLocator([-90, -86, -82, -78, -74, -70, -66, -62, -58, -52])
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.ylabel_style = {'size': 15, 'color': 'gray'}

    plt.savefig(KType+'_'+str(dxdy)+'_'+File_pdf)

    np.savetxt("Krig_"+KType+"_"+str(dxdy)+'_'+File_txt,z1,fmt='%.1f')
    #np.savetxt('lon'+File_txt,grid_lon,fmt='%.8f')
    #np.savetxt('lat'+File_txt,grid_lat,fmt='%.8f')

    print('Control Grid: ' + str(Control_Grid) + '$^\circ$' + ' Control Points: ' + str(N_Cpts) + '\n' + 'Ord. Kriging Type: ' + KType + ' Kriging Grid: ' + str(dxdy) + '$^\circ$')
    print(zzz)
    print("------------------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------------------")
    print("------------------------------------------------------------------------------------------------")

