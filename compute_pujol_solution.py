#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
from pdb import set_trace as bp
import pickle
from obspy.geodetics.base import gps2dist_azimuth, kilometer2degrees, degrees2kilometers
import pandas as pd
from obspy.core.utcdatetime import UTCDateTime
from functools import partial
from scipy import interpolate
from numpy import linalg as LA
from multiprocessing import get_context

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.optimize import least_squares
from multiprocessing import get_context
from pyproj import Geod
from pyproj import Proj

def compute_analytical_travel_times(observations, cs, x):
    """
    Compute travel time  
    x[0]: UTM longitude impact
    x[1]: UTM latitude impact
    x[2]: time of impact
    x[3]: trajectory azimuth angle
    x[4]: trajectory deflection angle
    x[5]: meteor velocity
    """

    ## Determine unit vector along trajectory
    u_x, u_y = np.sin(np.radians(90. - x[4])) * np.sin(np.radians(x[3])), np.sin(np.radians(90. - x[4])) * np.cos(np.radians(x[3]))
    uu = np.array([u_x, u_y, np.sin(np.radians(x[4]))])
    uu = np.repeat(uu[None,...], observations.shape[0], axis=0)
    
    ## Determine impact-station vector
    bb = observations[['x', 'y']].values
    bb = np.c_[bb, np.zeros((observations.shape[0],))]
    bb[:, 0] -= x[0]
    bb[:, 1] -= x[1]
    
    ## Determine distance between impact location and source location along trajectory for a given station
    dt = abs(bb[:, 0] * uu[:, 0] + bb[:, 1] * uu[:, 1] + bb[:, 2] * uu[:, 2])
    
    ## Determine distance between station to trajectory
    dp = np.sqrt( bb[:, 0]**2 + bb[:, 1]**2 + bb[:, 2]**2 - dt**2 )
    
    ## Compute travel time
    beta = np.arcsin( cs / x[5] )
    t = x[2] + (1/x[5]) * (-dt + dp/np.tan(beta))
    
    return t

def misfit_trajectory_function(observations, cs, x):
    """
    Compute travel time misfit from all stations to trajectory 
    """
    
    t = compute_analytical_travel_times(observations, cs, x)
    
    return np.sum( abs(t - observations.t.values) )
    
def compute_one_set_of_parameters(list_detections, parameter_space_all, idx):

    """
    Perform grid search over a specific subset of input parameters
    """
    
    ## Select appropriate subarray
    parameter_space = parameter_space_all[idx[0]:idx[1], :]
    
    ## Loop over each misfit
    misfits = pd.DataFrame()
    for i in range(parameter_space.shape[0]):
        x  = parameter_space[i, :].tolist()[:-1]
        cs = parameter_space[i, :].tolist()[-1]
        
        misfit = {
            'err': misfit_trajectory_function(list_detections, cs, x),
            'cs': cs,
            'lon': x[0],
            'lat': x[1],
            't0': x[2],
            'az': x[3],
            'deflection': x[4],
            'meteor_velocity': x[5]
        }
        misfits = misfits.append( [misfit] )
        
    return misfits
  
def deploy_on_CPUs(min_y, max_y, min_x, max_x, min_t0, max_t0, min_az, max_az, 
                   min_deflection, max_deflection, min_vel, max_vel, min_acous, max_acous,
                   nb_points_sample = 50000, nb_points_acous = 20, nb_CPU=16):

    ## Grid search
    np.random.seed(1)
    
    lats = np.random.uniform(low=min_y, high=max_y, size=nb_points_sample)
    lons = np.random.uniform(low=min_x, high=max_x, size=nb_points_sample)
    t0s  = np.random.uniform(low=min_t0, high=max_t0, size=nb_points_sample)
    azs  = np.random.uniform(low=min_az, high=max_az, size=nb_points_sample)
    deflections = np.random.uniform(low=min_deflection, high=max_deflection, size=nb_points_sample)
    vels = np.random.uniform(low=min_vel, high=max_vel, size=nb_points_sample)
    parameter_space = np.c_[lons, lats, t0s, azs, deflections, vels]
    parameter_space = np.tile(parameter_space.T, nb_points_acous).T
    
    acoustic_vels = np.random.uniform(low=min_acous, high=max_acous, size=nb_points_acous)
    acoustic_vels = np.repeat(acoustic_vels, nb_points_sample, axis=0)
    parameter_space = np.c_[parameter_space, acoustic_vels]
    
    partial_compute_one_set_of_parameters = partial(compute_one_set_of_parameters, list_detections, parameter_space)
    N = min(nb_CPU, nb_points_sample)
    ## If one CPU requested, no need for deployment
    if N == 1:
        misfit = partial_compute_one_set_of_parameters([0, parameter_space.shape[0]])
        bp()

    ## Otherwise, we pool the processes
    else:
        step_idx =  parameter_space.shape[0]//N
        list_of_lists = [[i*step_idx, (i+1)*step_idx] for i in range(N)]
        list_of_lists[-1][-1] = parameter_space.shape[0]
        
        with get_context("spawn").Pool(processes = N) as p:
            results = p.map(partial_compute_one_set_of_parameters, list_of_lists)

        misfit = pd.DataFrame()
        for result in results:
            misfit = misfit.append( result )
            
        misfit.reset_index(drop=True, inplace=True)
    
    misfit = misfit.sort_values(by='err')   
     
    return misfit
  
def plot_isochrone_maps(x, cs, lon_min, lon_max, lat_min, lat_max, proj_model, ax=None, nb_points = 100, use_latlon=False):

    """
    Plot travel time isochrones from meteor impact location
    """

    ## Build station map coordinates
    observations = pd.DataFrame()
    lon = np.linspace(lon_min, lon_max, nb_points)
    lat = np.linspace(lat_min, lat_max, nb_points)
    LAT, LON = np.meshgrid(lat, lon)
    LAT = LAT.ravel()
    LON = LON.ravel()
    observations['lon'] = LON
    observations['lat'] = LAT
    if not use_latlon:
        observations['x'], observations['y'] = observations.lon.values, observations.lat.values
    else:
        observations['x'], observations['y'] = proj_model(observations.lon.values, observations.lat.values)
    
    #import test_module; from importlib import reload 
    #lon, lat = 121., 25.
    #xx, yy = proj_taiwan(lon, lat)
    #x = [xx, yy, 225., 303., 70., 20.e3]
    #cs = 320.
    ## Compute corresponding arrival times
    times = compute_analytical_travel_times(observations, cs, x)
    
    ## Plot isochrone maps
    bounds = np.linspace(times.min()-x[2], times.max()-x[2], 7)
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256, extend='both')
    
    if ax == None:
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True);
        
    cmap = sns.color_palette("flare_r", as_cmap=True)
    if use_latlon:
        sc = ax.pcolormesh(lon, lat, times.reshape(nb_points, nb_points)-x[2], norm=norm, cmap=cmap)
    else:
        sc = ax.pcolormesh(observations['x'].values.reshape(nb_points, nb_points)/1e3, observations['y'].values.reshape(nb_points, nb_points)/1e3, times.reshape(nb_points, nb_points)-x[2], norm=norm, cmap=cmap)

    axins = inset_axes(ax, width="100%", height="4%", loc='lower left', bbox_to_anchor=(0., 1.02, 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    cb = plt.colorbar(sc, cax=axins, orientation='horizontal')
    cb.ax.set_xlabel('Travel time (s)', labelpad=10, fontsize=10)
    cb.ax.xaxis.tick_top()
    cb.ax.xaxis.set_label_position('top')
    for tick in cb.ax.get_xticklabels():
        tick.set_rotation(-90.)
  
def plot_results(dir_figures, list_detections, misfit, proj_model, ref_date, use_latlon=False, nb_points=100, l_trajectory=10000.):

    """
    Plot true arrivals along with a model solution
    """
    
    ## Setup figure
    fig  = plt.figure() 
    grid = fig.add_gridspec(3, 3)
    axs = []
    cmap = sns.color_palette("flare_r", as_cmap=True)
    
    axs.append( fig.add_subplot(grid[:, 0]) )
    axs.append( fig.add_subplot(grid[:, 1], sharey=axs[-1], sharex=axs[-1]) )
    
    vmin, vmax = misfit.err.min(), misfit.err.max()
    cols = misfit.loc[:, ~misfit.columns.isin(['err', 'cs'])].columns.tolist()
    simu = misfit.iloc[0]
        
    x  = simu[cols].values
    cs = simu['cs']
    
    ## Extract coordinates trajectory
    ts = compute_analytical_travel_times(list_detections, cs, x)
    if use_latlon:
        lon, lat = proj_model(x[0], x[1], inverse=True)
        wgs84_geod = Geod(ellps='WGS84')
        first_point = wgs84_geod.fwd(lon, lat, x[3]-180., l_trajectory)
    else:
        first_point = [x[0] + l_trajectory*np.sin(np.radians(x[3]-180.)), x[1] + l_trajectory*np.cos(np.radians(x[3]-180.)), l_trajectory*np.sin(np.radians(x[4])) ]

    if use_latlon:
        sc = axs[0].scatter(list_detections.lon, list_detections.lat, c=list_detections.t, vmin=list_detections.t.min(), vmax=list_detections.t.max()); 
        axs[1].scatter(list_detections.lon, list_detections.lat, c=ts, vmin=list_detections.t.min(), vmax=list_detections.t.max()); 
        axs[1].scatter([lon], [lat], marker='x', c=[simu.err], vmin=vmin, vmax=vmax, cmap=cmap); 
        axs[1].plot([lon, first_point[0]], [lat, first_point[1]], color='grey'); 
    else:
        sc = axs[0].scatter(list_detections.x/1e3, list_detections.y/1e3, c=list_detections.t, vmin=list_detections.t.min(), vmax=list_detections.t.max()); 
        plot_isochrone_maps(x, cs, min(list_detections.x.min(), x[0]), max(list_detections.x.max(), x[0]), min(list_detections.y.min(), x[1]), max(list_detections.y.max(), x[1]), proj_model, ax=axs[1], nb_points = nb_points, use_latlon=use_latlon)
        axs[1].scatter(list_detections.x/1e3, list_detections.y/1e3, c=ts, vmin=list_detections.t.min(), vmax=list_detections.t.max()); 
        axs[1].scatter(x[0]/1e3, x[1]/1e3, marker='x', c='tab:red', vmin=vmin, vmax=vmax, cmap=cmap); 
        axs[1].plot([x[0]/1e3, first_point[0]/1e3], [x[1]/1e3, first_point[1]/1e3], color='red', alpha=0.5); 
    
    axins = inset_axes(axs[0], width="100%", height="4%", loc='lower left', bbox_to_anchor=(0., 1.02, 1, 1.), bbox_transform=axs[0].transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    cb = plt.colorbar(sc, cax=axins, orientation='horizontal')
    cb.ax.set_xlabel('Arrival time (s)\nfrom ' + ref_date.strftime('%Y-%m-%d %H:%M:%S'), labelpad=10, fontsize=10)
    cb.ax.xaxis.tick_top()
    cb.ax.xaxis.set_label_position('top')
    plt.setp(axs[1].get_yticklabels(), visible=False)
    axs[0].set_xlabel('x (km)')
    axs[0].set_ylabel('y (km)')
    
    ## Plot probability distribution
    axs.append( fig.add_subplot(grid[0, 2]) )
    misfit_ = misfit.sort_values(by='cs')
    axs[-1].plot(misfit_.cs, misfit_.err/list_detections.shape[0])
    axs[-1].yaxis.tick_right()
    axs[-1].yaxis.set_label_position('right')
    axs[-1].set_title('Average error (s) vs $c_s$')
    
    ## Plot other properties
    time = ref_date + pd.to_timedelta(x[2], unit='s')
    str_description = 'Meteor vel.:\n' + str(np.round(x[5]/1e3, 2)) + ' km/s\n\nTime impact:\n' + time.strftime('%Y-%m-%d %H:%M:%S') + ' s' + '\n\nAzimuth:\n' + str(np.round(x[3],2)) + ' $^\circ$' + '\n\nDeflection:\n' + str(np.round(x[4],2)) + ' $^\circ$'
    axs[-1].text(0., -0.5, str_description, ha='left', va='top', transform=axs[-1].transAxes, fontsize=10.)
    
    fig.subplots_adjust(top=0.8)
    
    plt.savefig(dir_figures + 'inversion_Pujol_{ref_date}.pdf'.format(ref_date=time.strftime('%Y-%m-%dT%H.%M.%S.%f')[:-3]))
  
##########################
if __name__ == '__main__':
  
    ## Search parameters
    nb_points_sample = 100000 # Nb random sample from parameter space to test
    nb_points_acous  = 15 # Nb acoustic celerities to test per sample
    nb_CPU = 20
    dir_figures = '/staff/quentin/Documents/Projects/2021_meteor_reconstruction/figures_mars/'
    name_misfit_file_output = 'misfit_results_mars.csv'
    
    ## Mars meteor input parameters
    min_t0, max_t0 = -300., 300.
    min_az, max_az = 0., 350.
    min_deflection, max_deflection = 35., 70.
    min_range, max_range = 89.5*1e3, 102.*1e3
    min_vel, max_vel = 10.*1e3, 20.*1e3
    min_acous, max_acous = 200., 250.
    min_acous_surface, max_acous_surface = 238., 258.
    azimuth = 113.
    min_x, min_y = np.sin(np.radians(azimuth)) * min_range, np.cos(np.radians(azimuth)) * max_range
    max_x, max_y = np.sin(np.radians(azimuth)) * max_range, np.cos(np.radians(azimuth)) * min_range
    min_t0, max_t0 = -max_range / min_acous_surface, -min_range / max_acous_surface
    
    ## Reference time from which the impact time will be calculated 
    ref_date = pd.to_datetime('2021-09-01T12:00:00')
    
    ## Build list of observations as dataframe
    list_detections = pd.DataFrame()
    loc_dict = {
        'station': 'insight-0',
        't': 0., # (s)
        'x': 0., # (m)
        'y': 0., # (m)
    }
    list_detections = list_detections.append( [loc_dict] )
    
    ## Grid search
    misfit = deploy_on_CPUs(min_y, max_y, min_x, max_x, min_t0, max_t0, min_az, max_az, 
                   min_deflection, max_deflection, min_vel, max_vel, min_acous, max_acous,
                   nb_points_sample=nb_points_sample, nb_points_acous=nb_points_acous, nb_CPU=nb_CPU)
    misfit.reset_index(drop=True, inplace=True)
    misfit.to_csv(dir_figures + name_misfit_file_output, header=True, index=False)
    
    ## Plot results
    for id in range(5): 
        all_cs_from_id = misfit.loc[(misfit.lat == misfit.iloc[id].lat) & (misfit.lon == misfit.iloc[id].lon), :]
        plot_results(dir_figures, list_detections, all_cs_from_id, None, ref_date, use_latlon=False, nb_points=100, l_trajectory=10000.)
    bp()
