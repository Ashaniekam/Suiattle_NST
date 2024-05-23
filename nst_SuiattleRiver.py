# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:04:49 2024

@author: longrea
"""

"""
Spyder Editor

This is the code for my thesis on the Suiattle River

@author: longrea, pfeiffea

Goal: Sort out new Suiattle shapefile issues

"""
 #imports needed for code
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from landlab.io.native_landlab import load_grid, save_grid
import tempfile
from matplotlib.ticker import LogFormatter
import numpy as np
import pandas as pd
import os
import random
import pathlib
import xarray as xr
import time as model_time
import scipy.constants
from landlab.components import (
     FlowDirectorSteepest,
     NetworkSedimentTransporter,
     SedimentPulserEachParcel,
     SedimentPulserAtLinks,
     FlowAccumulator,
     BedParcelInitializerDepth,
 )
from landlab.data_record import DataRecord
from landlab.io import read_shapefile
from landlab.grid.network import NetworkModelGrid
from landlab.plot import graph
from landlab.plot import plot_network_and_parcels
from landlab.plot.network_sediment_transporter import plot_network_and_parcels

from landlab.components.network_sediment_transporter.aggregate_parcels import (
     aggregate_parcels_at_link_mean,
     aggregate_parcels_at_link_sum,
     aggregate_parcels_at_link_count,
 )

# Define a dictionary to store data for each variable for each scenario
scenario_data = {
    'elevation_change': [[] for _ in range(3)],  # Three scenarios
    'percent_active_pulse': [[] for _ in range(3)],
    'D_mean_pulse_each_link': [[] for _ in range(3)],
    'volume_pulse_at_each_link': [[] for _ in range(3)],
    'percent_pulse_remain': [[] for _ in range(3)],
    # Add other variables here
}
    

def run_scenario(scenario):
    
    OUT_OF_NETWORK = NetworkModelGrid.BAD_INDEX - 1
    
    timesteps = 2002
    pulse_time = 1002
    time_since_ss = timesteps- pulse_time
    time_array= np.arange(1, timesteps + 1)



     # #### version #######
    version = "new"
        # XXX TIME in the model

     # # XXX Channel Morphology data
     #read in shapefiles
    shp_file = os.path.join(os.getcwd (), ("Suiattle_river_"+version+".shp"))
    points_shapefile = os.path.join(os.getcwd(), ("Suiattle_nodes_"+version+".shp"))

    #create the grid
    grid = read_shapefile(
         shp_file,
         points_shapefile = points_shapefile,
         node_fields = ["usarea_km2", "elev_m"],
         link_fields = ["usarea_km2", "Length_m", "Slope", "Width"],
         link_field_conversion = {"usarea_km2": "drainage_area", "Slope":"channel_slope", "Width": "channel_width", "Length_m":"reach_length", }, #, "Width": "channel_width";  how to read in channel width
         node_field_conversion = {"usarea_km2": "drainage_area", "elev_m": "topographic__elevation"},
         threshold = 0.1,
         )

     #discharge from  DSHVM Excel file
    q_dataframe = pd.read_excel(r"C:\Users\longrea\OneDrive - Western Washington University\Thesis\1_Code\Suiattle Code v2\Discharge from DHSVM Suiattle.xlsx", sheet_name = 'Discharge (2% exceedance)', header = 0)
    discharge = q_dataframe['Q'].values


     #Using drainage area for links and nodes to create ordering indices
    area_link = grid.at_link["drainage_area"]
    area_node = grid.at_node["drainage_area"]

    index_sorted_area_link = (area_link).argsort()
    index_sorted_area_node = (area_node).argsort()

    #Sorted Drainage Area
    area_link = area_link[index_sorted_area_link]
    area_node = area_node[index_sorted_area_node]


    #Sorted Topographic Elevation and Bedrock Elevation
    topo = grid.at_node['topographic__elevation'].copy()
    topo[topo== 776.32598877] += 3 #attemp to smooth the grid to get rid of bottleneck
    topo = topo[index_sorted_area_node]
    grid.at_node["bedrock__elevation"] = topo.copy()

    #Sorted Width
    width = grid.at_link["channel_width"]
    width = width[index_sorted_area_link]
    initial_width = width.copy()

    #Sorted Slope
    slope = grid.at_link["channel_slope"]
    slope = slope[index_sorted_area_link]
    initial_slope= slope.copy()

    #Sorted Length
    length = grid.at_link["reach_length"]
    length = length[index_sorted_area_link]
    cum_sum_upstream = np.cumsum(length)
    grid.at_link["dist_upstream"]= cum_sum_upstream
    dist_upstream = grid.at_link["dist_upstream"]/1000
    dist_upstream_nodes = np.insert(dist_upstream, 0, 0.0)

    number_of_links = grid.number_of_links

    
    if scenario == 1:
        scenario_num = "No abrasion"
        model_state = "Testing_Density_Run no_abrasion"
        pass
    elif scenario == 2:
        scenario_num = "SHRS proxy abrasion"
        model_state = "Testing_Density_Run SHRS proxy"
        pass
    else: 
        scenario_num = "Double SHRS proxy abrasion"
        model_state = "Testing_Density_Run SHRSAbrasion_correction2"

   
    
    #to calculate flow depth

    Mannings_n = 0.086 #0.1734 #median calculated from Suiattle gages (units m^3/s)
    grid.at_link["flow_depth"] = ((Mannings_n*discharge)/ ((slope**0.5)*width))**0.6

    depth = grid.at_link["flow_depth"].copy()

    rho_water = 1000
    rho_sed = 2650
    gravity = 9.81
    tau = rho_water * gravity * depth * slope

        
    #start for scaling sediment feed by drainage area
    #calculates the difference between the upstream and downstream dranage area for each link
    diff_area_at_each_link =  []
    for d in range ((len(area_link)-1)):
        diff_area = area_link[d] - area_link[d+1]
        diff_area_at_each_link.append(diff_area)


    vol_of_channel = width*length*depth

    


    #Plotting river characteristics
    #FLOW DEPTH
    plt.figure()
    plt.plot(dist_upstream, depth, '-' )
    plt.title("Flow Depth")
    plt.ylabel("Flow depth (m)")
    plt.xlabel("Distance downstream (km)")
    plt.show()

    # #Channel Width
    plt.figure()
    plt.plot(dist_upstream, width, '-' )
    plt.title("Channel Width")
    plt.ylabel("Channel Width (m)")
    plt.xlabel("Distance downstream (km)")
    plt.show()
    
    # plt.figure()
    # plt.plot(dist_upstream, grid.at_link["channel_slope"], '-' )
    # plt.title("Starting channel Slope")
    # plt.ylabel("Slope")
    # plt.xlabel("Distance downstream (km)")
    # plt.show()

    # #TAU
    #rho = 1000
    # gravity = 9.81
    # tau = rho * gravity * depth * slope
    # plt.figure()
    # plt.plot(dist_upstream, tau)
    # plt.title("Tau")
    # plt.ylabel("Shear Stress")
    # plt.xlabel("Distance downstream (km)")
    # plt.show()

    plt.figure()
    plt.plot(dist_upstream, slope, '-' )
    plt.title("Channel Slope")
    plt.ylabel("Slope")
    plt.xlabel("Distance downstream (km)")
    plt.show()
    
    #variables needed for plots
    # XXX
    n_recycled_parcels = np.empty(timesteps)
    n_pulse_left = np.empty(timesteps)
    n_pulse_on = np.empty(timesteps)
    d_recycled_parcels = np.empty(timesteps)
    time_elapsed = np.empty(timesteps)
    count_recyc = np.empty(timesteps)
    timestep_array = np.arange(timesteps)
    sed_total_vol = np.empty([grid.number_of_links,timesteps])
    sediment_active_percent = np.empty([grid.number_of_links,timesteps])
    active_d_mean= np.empty([timesteps,grid.number_of_links])
    num_active_pulse= np.empty([grid.number_of_links,timesteps])
    num_active= np.empty([grid.number_of_links,timesteps])
    volume_pulse_at_each_link = np.empty([grid.number_of_links,timesteps])
    volume_at_each_link = np.empty([grid.number_of_links,timesteps])
    D_mean_pulse_each_link= np.empty([grid.number_of_links,timesteps])
    D_mean_each_link= np.empty([grid.number_of_links,timesteps])
    num_pulse_each_link= np.empty([grid.number_of_links,timesteps], dtype=int)
    num_each_link= np.empty([grid.number_of_links,timesteps], dtype=int)
    num_total_parcels_each_link= np.empty([grid.number_of_links,timesteps], dtype=int)
    num_count_recyc_parc= np.empty([grid.number_of_links,timesteps], dtype=int)
    jon_W= np.empty([grid.number_of_links,timesteps])
    transport_capacity= np.empty([grid.number_of_links,timesteps])
    num_recyc_each_link= np.empty([grid.number_of_links,timesteps], dtype=int)
    change_D= np.empty([grid.number_of_links,timesteps])
    link_len = grid.length_of_link


    Elev_change = np.empty([grid.number_of_nodes,timesteps]) # 2d array of elevation change in each timestep 
    before_ss_Elev_change = np.empty([grid.number_of_nodes,timesteps]) # 2d array of elevation change in each timestep 
    ss_Elev_change = np.empty([grid.number_of_nodes,time_since_ss]) # use cap first letter to denote 2d 

    count_recyc_parc = []

    #creating sediment parcels in the DATARECORD

    slope_depend_Shields = 0.15* slope**0.25
    tau_c_multiplier = 2.4 #to change grain size
    tau_c_50 = slope_depend_Shields
    median_number_of_starting_parcels = 100

    initialize_parcels = BedParcelInitializerDepth(
        grid,
        flow_depth_at_link = depth,
        tau_c_50 = slope_depend_Shields,
        tau_c_multiplier = tau_c_multiplier,
        median_number_of_starting_parcels = 100,
        extra_parcel_attributes = ["source", "recycle_destination"]
        )

    parcels = initialize_parcels()

    #Assigning the extra parcel attributes listed above
    parcels.dataset["source"].values = np.full(parcels.number_of_items, "initial_bed_sed")
    #will be used to track the parcels that are recycled
    parcels.dataset["recycle_destination"].values = np.full(parcels.number_of_items, "not_recycled_ever")
    #parcels.dataset["change_D"].values = np.zeros(np.shape(parcels.number_of_items))
    grain_size= parcels.dataset.D.values
    sorted_grain_sizes = np.sort(grain_size)

    #calculate D50 
    d_50 = (rho_water * depth * slope) / ((rho_sed - rho_water) * tau_c_multiplier * tau_c_50)

    # calculation of the initial volume of sediment on each link
    parcels_start_bool = parcels.dataset["time"] == parcels.dataset.time[0].values

    initial_vol_on_link = np.empty(number_of_links, dtype = float)
    aggregate_parcels_at_link_sum(
        initial_vol_on_link,
        number_of_links,
        parcels.dataset.element_id.values[:, -1].astype(int),
        len(parcels.dataset.volume.values[:, -1]),
        parcels.dataset.volume.values[:, -1],
    )

    # Plot and calculation for TAU_star
    rho_sed = 2650

    D50_each_link = parcels.calc_aggregate_value(
        xr.Dataset.median, "D", at="link", fill_value=0.0)
    tau_star = tau/((rho_sed - rho_water) * gravity * D50_each_link)
    plt.figure()
    plt.title("Tau_star")
    plt.plot(dist_upstream, tau_star)
    plt.ylabel("Shield Stress")
    plt.xlabel("Distance upstream (km)")
    plt.show()

    #find line of best fit
    a, b = np.polyfit(dist_upstream, D50_each_link, 1)
    str_tau_c_multiplier = str(tau_c_multiplier)
    plt.figure()
    plt.scatter(dist_upstream, D50_each_link)
    plt.plot(dist_upstream, a*dist_upstream+b)  
    plt.ylabel("D50 each link")
    plt.title("d50 each link (tau_c_multiplier = " + str_tau_c_multiplier +")")
    plt.xlabel("Distance upstream (km)")
    plt.show()

    a, b = np.polyfit(dist_upstream, d_50, 1)
    str_tau_c_multiplier = str(tau_c_multiplier)
    plt.figure()
    plt.scatter(dist_upstream, d_50)
    plt.plot(dist_upstream, a*dist_upstream+b)  
    plt.ylabel("D50")
    plt.title("d50 (tau_c_multiplier = " + str_tau_c_multiplier +")")
    plt.xlabel("Distance upstream (km)")
    plt.show()


    dt = 60*60*24*1  # len of timesteps (1 day)
    #dt = 60*60*24*7  # len of timesteps (seconds) 7 days
    len_dt = dt/86400

    n_lines = 10 # note: # timesteps must be divisible by n_lines with timesteps%n_lines == 0
    color = iter(plt.cm.viridis(np.linspace(0, 1, n_lines+1)))
    c = next(color)

    # flow direction
    fd = FlowDirectorSteepest(grid, "topographic__elevation")
    fd.run_one_step()

    # initialize the networksedimentTransporter
    bed_porosity=0.3

    nst = NetworkSedimentTransporter(
        grid,
        parcels,
        fd,
        bed_porosity=bed_porosity,
        g=9.81,
        fluid_density=1000,
        transport_method="WilcockCrowe",
        active_layer_method = "Constant10cm",
        active_layer_d_multiplier= 0.5,
    )
    
    bed_area = length * width #(m^2)
    #effective_pulse_depth = (num_pulse_parcels*pulse_parcel_vol)/(np.sum(bed_area[15:37]))

    Pulse_cum_transport = []

    thickness = ((np.nanmedian(parcels.dataset["volume"].values))*median_number_of_starting_parcels)/(grid.at_link["reach_length"]*width)
    num_parcels = parcels.number_of_items #total number of parcels in network

    
    # #XXX run model in time      
    for t in range(0, (timesteps*dt), dt):
        #by index of original sediment
        start_time = model_time.time()
        mask_ispulse = parcels.dataset.source == 'pulse'

        total_number_of_parcels = parcels.dataset.element_id.values[:, -1].astype(int)
        
        #boolean mask of parcels that have left the network
        mask1 = parcels.dataset.element_id.values[:,-1] == OUT_OF_NETWORK
        #boolean mask of parcels that are initial bed parcels (no pulse parcels)
        mask2 = parcels.dataset.source == "initial_bed_sed" 
        
        bed_sed_out_network = parcels.dataset.element_id.values[mask1 & mask2, -1] == -2
        
        #index of the bed parcels that have left the network
        index_initial_bed_sed = np.where(bed_sed_out_network == True) 
        
        #number of parcels that will be recycled/ similar to n_recycled_parcels below
        num_bed_parcels = np.size(index_initial_bed_sed)

        # Calculate the total sum of area_link
        total_area = sum(area_link)

        # Calculate the proportions based on drainage area
        proportions = [area / total_area for area in area_link]

        # Calculate the number of parcels for each link based on proportions
        num_recycle_bed = [int(proportion * num_bed_parcels) for proportion in proportions]

        # Calculate the remaining parcels not accounted for by using int above
        remaining_bed_sed = num_bed_parcels - sum(num_recycle_bed)

        # Calculate the number of additional parcels to add to each proportion
        additional_bed_sed = np.round(np.array(proportions) * remaining_bed_sed).astype(int)

        # Distribute the remaining parcels to the proportions with the largest values
        while remaining_bed_sed > 0:
            max_index = np.argmax(additional_bed_sed)
            additional_bed_sed[max_index] -= 1
            num_recycle_bed[max_index] += 1
            remaining_bed_sed -= 1

        # Initialize an empty list to store the indices
        indices = []

        # Generate indices for each proportion
        for i, num in enumerate(num_recycle_bed):
            indices.extend([i] * num)

        indices = np.squeeze(indices)
        
        #current_sum = np.sum(num_recycle_bed)
        #count_recyc_parc.append(num_recycle_bed)
        
        
        
        #adding sediment by recycling
     
        n_recycled_parcels[np.int64(t/dt)]=np.sum(parcels.dataset.element_id.values[:,-1] == OUT_OF_NETWORK,-1)
        d_recycled_parcels[np.int64(t/dt)]=np.mean(parcels.dataset.D.values[parcels.dataset.element_id.values[:,-1] == OUT_OF_NETWORK,-1])
        
        
        
        # assign the new starting link
        for bed_sed in index_initial_bed_sed:
            parcels.dataset.element_id.values[bed_sed, 0] = indices
            recyc_= np.where(parcels.dataset.element_id.values[bed_sed,0] == indices)
            
            for recycle_destination in indices:
                parcels.dataset["recycle_destination"].values[recyc_] = np.full(num_bed_parcels, "recycle_des %s" %recycle_destination)

                #recyc_element_ids, count_recyc_parc = np.unique(num_elements, return_counts=True)
                
        nst.run_one_step(dt)
        
        #making a pulse
        if t==dt*pulse_time: 
            print('making a pulse')
            print('t = ',t, ',timestep = ', t/dt)
            
            steady_state_elevation = grid.at_node['topographic__elevation'].copy()
            
            
            
            fan_thickness = np.array([6.25, 6.25, 5.75, 2.75]) #meters
            fan_location= np.array([7, 8, 9, 10])
            pulse_parcel_vol = 10 #volume of each parcel m^3
            pulse_volume = fan_thickness*length[fan_location]*width[fan_location] # m^3; volume of pulse that should be added to each link of the Chocolate Fan
            pulse_rock_volume = pulse_volume * (1-bed_porosity)
            num_pulse_parcels_by_vol = pulse_rock_volume/pulse_parcel_vol #number of parcels added to each link based on volume; 
            total_num_pulse_parcels = int(np.sum(num_pulse_parcels_by_vol))
            pulse_location = [index for index, freq in enumerate((num_pulse_parcels_by_vol).astype(int), start=7) for _ in range(freq)]
            random.shuffle(pulse_location)
            
            
            for fan in fan_location:
                alluvium_depth_fan= 2*pulse_volume/(np.sum(width[fan-1]* length[fan-1])+ width[fan + 1]*length[fan + 1])/ (1-bed_porosity)
            #pulse_location = [random.randrange(8, 13, 1) for i in range(num_pulse_parcels)] #code from nst_test.py: adding pulse randomly near the fan
            newpar_element_id = pulse_location
            newpar_element_id = np.expand_dims(newpar_element_id, axis=1)
            
            Pulse_cum_transport = np.ones([total_num_pulse_parcels,1])*np.nan
            new_starting_link = np.squeeze(pulse_location)
            
            np.random.seed(0)
            
            new_time_arrival_in_link = nst._time* np.ones(
                np.shape(newpar_element_id)) 
             
            new_volume = pulse_parcel_vol*np.ones(np.shape(newpar_element_id))  # (m3) the volume of each parcel
            
            new_source = ["pulse"] * np.size(
                newpar_element_id
            )  # a source sediment descriptor for each parcel
            
            new_active_layer = np.zeros(
                np.shape(newpar_element_id)
            )  # 1 = active/surface layer; 0 = subsurface layer
            
           
            
            
            
            
        
            
            new_location_in_link = np.random.rand(
                np.size(newpar_element_id), 1
            )
            #measured_alphas = alphamultiplier*3.0715*np.exp(-0.136*SHRS)
            field_data = pd.read_excel(r"C:\Users\longrea\OneDrive - Western Washington University\Thesis\1_Code\Suiattle Code v2\SuiattleFieldData_Combined20182019.xlsx", sheet_name = '4-SHRSdfDeposit', header = 0)
            SHRS = field_data['SHRS median'].values
            SHRS_MEAN = np.mean(SHRS)
            SHRS_STDEV = np.std(SHRS)

            SHRS_distribution = np.random.normal(SHRS_MEAN, SHRS_STDEV, (np.size(newpar_element_id)))


            measured_alphas = 3.0715*np.exp(-0.136*SHRS_distribution) # units: 1/km
            
            #new_density = 894.978992640976*np.log(distribution)-1156.7599235065895


            tumbler_2 = 2 * (3.0715*np.exp(-0.136*SHRS_distribution))
            tumbler_4 = 4 * (3.0715*np.exp(-0.136*SHRS_distribution))
            
            if scenario == 1:
                new_abrasion_rate = 0 * np.ones(np.size(newpar_element_id))
                new_density = 2650 * np.ones(np.size(newpar_element_id))  # (kg/m3) standard
                print('Scenario 1 pulse: no abrasion')
            elif scenario == 2:
                new_abrasion_rate = (measured_alphas/1000)* np.ones(np.size(newpar_element_id)) #0.3 % mass loss per METER
                new_density = 894.978992640976*np.log(SHRS_distribution)-1156.7599235065895
                print('Scenario 2 pulse: variable abrasion, SH proxy')
            elif scenario == 3: 
                new_abrasion_rate = (tumbler_4/1000)* np.ones(np.size(newpar_element_id)) #tumbler correction 4 testing!!!!
                new_density = 894.978992640976*np.log(SHRS_distribution)-1156.7599235065895
            else: 
                new_abrasion_rate = (tumbler_4/1000) * np.ones(np.size(newpar_element_id))
            
            
            SHRS_proxy = (measured_alphas/1000)* np.ones(np.size(newpar_element_id))
            Double_SHRS = (tumbler_2/1000)* np.ones(np.size(newpar_element_id))
            # volume_post_abrasion = pulse_parcel_vol * np.exp(nst._distance_traveled_cumulative[-num_pulse_parcels:]* (-new_abrasion_rate))
            # abrasion_percentage = (volume_post_abrasion/10)*100
                
            new_D= np.random.lognormal(np.log(0.09),np.log(1.5),np.shape(newpar_element_id))    
            # (m) the diameter of grains in each parcel
            #original::  new_D = np.random.lognormal(np.log(0.03 "D50"),np.log(3),np.shape(newpar_element_id))
            initial_d_pulse = new_D.copy()
            

            newpar_grid_elements = np.array(
                np.empty(
                    (np.shape(newpar_element_id)), dtype=object
                )
            )
            
            newpar_grid_elements.fill("link")
            
            new_parcels = {"grid_element": newpar_grid_elements,
                     "element_id": newpar_element_id}

            new_variables = {
                "starting_link": (["item_id"], new_starting_link),
                "abrasion_rate": (["item_id"], new_abrasion_rate),
                "density": (["item_id"], new_density),
                "source": (["item_id"], new_source),
                "time_arrival_in_link": (["item_id", "time"], new_time_arrival_in_link),
                "active_layer": (["item_id", "time"], new_active_layer),
                "location_in_link": (["item_id", "time"], new_location_in_link),
                "D": (["item_id", "time"], new_D),
                "volume": (["item_id", "time"], new_volume),
                
        
                }
            
            parcels.add_item(
                    time=[nst._time],
                    new_item = new_parcels,
                    new_item_spec = new_variables
            )
            

       
        if t > dt*pulse_time :
            Pulse_cum_transport = np.append(
                    Pulse_cum_transport,
                    np.expand_dims(
                            nst._distance_traveled_cumulative[-total_num_pulse_parcels:],
                            axis = 1),
                    axis = 1
                    )
        
        if t == parcels.dataset.time[0].values: # if we're at the first timestep

            avg_init_sed_thickness = np.mean(
                grid.at_node['topographic__elevation'][:-1]-grid.at_node['bedrock__elevation'][:-1])        
            grid.at_node['bedrock__elevation'][-1]=grid.at_node['bedrock__elevation'][-1]+avg_init_sed_thickness


            elev_initial = grid.at_node['topographic__elevation'].copy()
    
            
        thickness_at_link = volume_pulse_at_each_link[:,2]/grid.at_link["channel_width"]/grid.at_link["reach_length"]/(1-bed_porosity)    
        #if t%((timesteps*dt)/n_lines)==0:        
        
            # # PLOT 1: elevation change through time, on long profile
            
            # plt.figure("Tracking elev change")
            
            # elev_change = grid.at_node['topographic__elevation']-elev_initial
            # plt.plot(dist_upstream_nodes, elev_change, c=c)
            # plt.xlabel('Distance downstream (km)')
            # plt.title("Elevation change through time")
            # plt.ylabel('Elevation change (m)')
            
            

            #print('plotted at timestep',t/dt)
        #total_number_of_parcels_int = parcels.number_of_items    
        print("Model time: ", t/(60*60*24), "days passed (", t/dt, 'timesteps)')
        print('Elapsed:', (model_time.time() - start_time)/60 ,' minutes')
        
        
        # Populating arrays with data from this timestep. 
        sed_act_vol= grid.at_link["sediment__active__volume"][index_sorted_area_link] 
      
        sed_total_vol[:,np.int64(t/dt)] = grid.at_link["sediment_total_volume"][index_sorted_area_link]
        sediment_active_percent[:,np.int64(t/dt)]= sed_act_vol/sed_total_vol[:,np.int64(t/dt)]
        
        Elev_change[:,np.int64(t/dt)] = grid.at_node['topographic__elevation']-elev_initial
        
        
        #PLOT5: NUMBER OF ACTIVE PARCELS THROUGH TIME
        parcels_end = parcels.dataset["time"] == parcels.dataset.time[-1].values
        
        num_active_parcels_each_link = np.empty(number_of_links, dtype = float)
        aggregate_parcels_at_link_sum(
            num_active_parcels_each_link,
            number_of_links,
            parcels.dataset.element_id.values[:, -1].astype(int),
            len(parcels.dataset.active_layer.values[:, -1]),
            parcels.dataset.active_layer.values[:, -1],
        )
        
        
        sorted_num_active_parcels_each_link = num_active_parcels_each_link[index_sorted_area_link]

    #%%
        #PLOT 6: Fraction of active bed made up of pulse material

        mask_ispulse = parcels.dataset.source == 'pulse'
        mask = parcels.dataset["time"]== parcels.dataset.time[-1].values
        mask_active = parcels.dataset.active_layer[:,-1]==1 
        
        #transport capacity plot 
        active_parcels = parcels.dataset.element_id.values[mask_active,-1].astype(int) #array of active parcels' element IDs
        parcel_volume = parcels.dataset.volume.values[:, -1] #parcel volume
        active_layer_volume = nst._active_layer_thickness *width*length
        
        weighted_sum_velocity = 0
        #sum by element ID (nst._pvelocity[active_parcels]* parcel volume[active_parcels])
        for parcel_id in active_parcels:
            parcel_velocity = nst._pvelocity[parcel_id]
            parcel_vol = parcel_volume[parcel_id]
            weighted_sum_velocity += parcel_velocity * parcel_vol
            #total_weight += parcel_vol
            jon_W [:,np.int64(t/dt)]= weighted_sum_velocity/ length
        
        weighted_mean_velocity = weighted_sum_velocity / active_layer_volume #divide that by active layer volume 
        
        
        # Convert weighted-mean velocity to m/s
        transport_capacity [:,np.int64(t/dt)]= weighted_mean_velocity * nst._active_layer_thickness * width  #multiply by active layer thickness and link width 
        
        
       
        element_all_parcels = parcels.dataset.element_id.values[:, -1].astype(int)
        element_pulse= element_all_parcels[mask_ispulse]
         
        n_pulse_on[np.int64(t/dt)]=np.sum(parcels.dataset.element_id.values[mask_ispulse,-1] != OUT_OF_NETWORK,-1)
        n_pulse_left[np.int64(t/dt)]=np.sum(parcels.dataset.element_id.values[mask_ispulse,-1] == OUT_OF_NETWORK,-1)
         
        
        percent_pulse_remain = (n_pulse_on/(n_pulse_on + n_pulse_left))*100
        
        aggregate_parcels_at_link_sum(
            num_active_pulse[:,np.int64(t/dt)],
            number_of_links, 
            element_pulse,
            len(parcels.dataset.active_layer.values[mask_ispulse, -1]),
            parcels.dataset.active_layer.values[mask_ispulse, -1],
        )

        
        sorted_num_active_pulse = num_active_pulse[index_sorted_area_link]
        
        #for all parcels
        aggregate_parcels_at_link_sum(
            num_active[:,np.int64(t/dt)],
            number_of_links, 
            element_all_parcels,
            len(parcels.dataset.active_layer.values[:, -1]),
            parcels.dataset.active_layer.values[:, -1],
        )

        
        sorted_num_active = num_active[index_sorted_area_link]
        
        ## Plot 7: D_MEAN_ACTIVE through time
        active_d_mean [np.int64(t/dt),:]= nst.d_mean_active
        initial_d = active_d_mean[0]
        
        
        #plot8: Volume of pulse parcels through time
        
        aggregate_parcels_at_link_sum(
            volume_pulse_at_each_link[:,np.int64(t/dt)],
            number_of_links, 
            element_pulse,
            len(parcels.dataset.volume.values[mask_ispulse, -1]),
            parcels.dataset.volume.values[mask_ispulse, -1],
        )
        sorted_volume_pulse_at_each_link = volume_pulse_at_each_link[index_sorted_area_link]
        
        #plot 8 for all parcels
        aggregate_parcels_at_link_sum(
            volume_at_each_link[:,np.int64(t/dt)],
            number_of_links, 
            element_all_parcels,
            len(parcels.dataset.volume.values[:, -1]),
            parcels.dataset.volume.values[:, -1],
        )
        sorted_volume_at_each_link = volume_at_each_link[index_sorted_area_link]
        
        #PLOT 9: D mean of pulse at each link
        aggregate_parcels_at_link_mean(
            D_mean_pulse_each_link[:,np.int64(t/dt)], 
            number_of_links,
            element_pulse,
            len(parcels.dataset.D.values[mask_ispulse,-1]),
            parcels.dataset.D.values[mask_ispulse,-1],
            parcels.dataset.volume.values[:, -1],
        )
        sorted_D_mean_pulse_each_link = D_mean_pulse_each_link[index_sorted_area_link]
        
        #plot 9 for all parcels
        aggregate_parcels_at_link_mean(
            D_mean_each_link[:,np.int64(t/dt)], 
            number_of_links,
            element_all_parcels,
            len(parcels.dataset.D.values[:,-1]),
            parcels.dataset.D.values[:,-1],
            parcels.dataset.volume.values[:, -1],
        )
        sorted_D_mean_each_link = D_mean_each_link[index_sorted_area_link]
        
        # change_D_calc = parcels.dataset.D.values[mask_ispulse,-1] - new_D[:,0]
        # aggregate_parcels_at_link_mean(
        #     change_D[:,np.int64(t/dt)], 
        #     number_of_links,
        #     element_all_parcels,
        #     len(change_D_calc),
        #     change_D_calc,
        #     parcels.dataset.volume.values[:, -1],
        # )
        #PLOT 10: number of pulse parcel each link
        aggregate_parcels_at_link_count(
            num_pulse_each_link[:,np.int64(t/dt)],
            num_pulse_each_link.shape[0],
            element_pulse,
            len(element_pulse),
        )
        sorted_num_pulse_each_link =  num_pulse_each_link[index_sorted_area_link]
        
        #for all parcels
        aggregate_parcels_at_link_count(
            num_each_link[:,np.int64(t/dt)],
            num_each_link.shape[0],
            element_all_parcels,
            len(element_all_parcels),
        )
        sorted_num_each_link =  num_each_link[index_sorted_area_link]
        
        
        # Plot #: number of total parcels each link
        aggregate_parcels_at_link_count(
             num_total_parcels_each_link[:,np.int64(t/dt)],
             num_total_parcels_each_link.shape[0],
             total_number_of_parcels,
             len(total_number_of_parcels),
             
         )
        sorted_num_total_parcels_each_link = num_total_parcels_each_link[index_sorted_area_link]
        
        # initial_vol_on_link_pulse = np.empty([grid.number_of_links,timesteps], dtype=int)
        # aggregate_parcels_at_link_sum(
        #     initial_vol_on_link_pulse[:,np.int64(t/dt)],
        #     number_of_links,
        #     element_pulse,
        #     len(parcels.dataset.volume.values[mask_ispulse, -1]),
        #     parcels.dataset.volume.values[mask_ispulse, -1],
        # )

        # final_vol_on_link_pulse = np.empty([grid.number_of_links,timesteps], dtype=int)
        # aggregate_parcels_at_link_sum(
        #     final_vol_on_link_pulse[:,np.int64(t/dt)],
        #     number_of_links,
        #     element_pulse,
        #     len(parcels.dataset.volume.values[mask_ispulse, -1]),
        #     parcels.dataset.volume.values[mask_ispulse, -1],
        # )
    #saving dataset to Excel
    # parcels_dataframe= parcels.dataset.to_dataframe()
    # last_timestep = parcels.dataset["time"].max() 
    # last_timestep_data = parcels_dataframe[parcels.dataset['time'] == last_timestep]  # Filter rows for the last timestep
    # last_timestep_data.to_csv('Parcels_data_last_timestep.csv', sep=',', index=True) 
    
    parcels_dataframe = parcels.dataset.to_dataframe()
    #Save the entire dataset
    parcels_dataframe.to_csv('Parcels_data_scenario.csv', sep=',', index=True)

    # Read the big Excel file into a DataFrame
    parcels_dataframe = pd.read_csv('Parcels_data_scenario.csv')

    # Last timestep in the saved excel file
    max_id = parcels_dataframe['time'].max()

    # Filter the DataFrame to include only rows from the last timestep
    filtered_data = parcels_dataframe[parcels_dataframe['time'] == max_id]

    # Save the filtered data to a new Excel file
    filtered_data.to_excel('Filtered_Parcels_data_scenario.xlsx', index=False)

    
    plt.figure()
    plt.plot(area_link, num_recycle_bed, c=c)
    plt.title("count_recyc_parc (Scenario = %i)" %scenario)
    plt.xlabel("Drainage area")
    plt.ylabel("Number of recycled parcels")
    #plt.savefig('count_recyc_parc (Scenario'+scenario_num+ ').png')

    

    # plt.figure()
    # plt.plot(count_recyc)
    # plt.ylabel('Number of recycled parcels')
    # plt.xlabel('Drainage Area')
    # plt.show()

    #print("This runs to 1000 timesteps to check if A changes from not being empty")

    # #XXX Saving a text file of model characteristics
    new_dir_name = model_state
    new_dir = pathlib.Path('C:/Users/longrea/OneDrive - Western Washington University/Thesis/1_Code/Results and plots', new_dir_name)
    new_dir.mkdir(parents=True, exist_ok=True)
    # Converting the time in seconds to a timestamp
    c_ti = model_time.ctime(os.path.getctime("NST_Suiattle_April_26_current.py"))
    text_file = os.path.join(new_dir, 'Suiattle_run_characteristics ('+scenario_num+ ').txt')   
    file = open('Suiattle_run_characteristics'+scenario_num+ ').txt', 'w')
    model_characteristics = ["This script was created at %s" %c_ti, "This code is running for scenario %s" %scenario,  "This is the tau_c_multiplier %s" %tau_c_multiplier, 
                              "This is the number of timesteps %s" %timesteps, "The number of days in each timestep is %s" %len_dt,
                              "The pulse is added at timestep %s" %pulse_time,
                              "The volume of each pulse parcel is %s" %pulse_parcel_vol, "This is the number of pulse parcels added to the network %s"
                              %total_num_pulse_parcels,
                              
                              "The current stage OF THE CODE IS TESTING NO ABRASION AND CONSTANT DENSITY VS SHRS DISTRIBUTION AND CORRECTION OF 4 WITH DISTRIBUTION OF DENSITY"]
    for line in model_characteristics:
    # file.write('%d' % link_len)
        file.write(line)
        file.write('\n')
    file.close()

    #save grid for plotting
    fname = os.path.join(os.getcwd (), 'Suiattle_scenario'+scenario_num+'.grid')
    grid.to_netcdf(fname, 'w')
        
    # %% ####   Plotting and analysis of results    ############  

    # Print some pertinent information about the model run
    print('mean grain size of pulse',np.mean(
        parcels.dataset.D.values[parcels.dataset.source.values[:] == 'pulse',-1]))

    plt.figure()
    for parcel_sizes in range(len(active_d_mean)):
        grain_size_change = initial_d- active_d_mean[parcel_sizes]
        plt.plot(grain_size_change)
    plt.show()

    # final volume of sediment on each link

    final_vol_on_link = np.empty(number_of_links, dtype = float)
    aggregate_parcels_at_link_sum(
        final_vol_on_link,
        number_of_links,
        element_all_parcels,
        len(parcels.dataset.volume.values[:, -1]),
        parcels.dataset.volume.values[:, -1],
    )
    sorted_final_vol_on_link = final_vol_on_link[index_sorted_area_link] 
    
    percent_active_pulse= num_active_pulse/ num_pulse_each_link
    sorted_percent_active_pulse = percent_active_pulse[index_sorted_area_link] 
    sorted_percent_active_pulse[sorted_percent_active_pulse == 0]= np.nan



    ## Adding elevation change as an at_link variable
    elev_change_at_link = ((final_vol_on_link - initial_vol_on_link) /
                            (grid.at_link["reach_length"]*grid.at_link["channel_width"]))  # volume change in link/ link bed area

    #links=np.arange(grid.number_of_links)
    grid.add_field("elevation_change", elev_change_at_link, at="link", units="m", copy=True, clobber=True)
    grid.add_field("change_D", grain_size_change, at="link", units="m", copy=True, clobber=True)
    #grid.add_field("grain_sizes", grain_size, at="link", units="m", copy=True, clobber=True)
    
    scenario_data['elevation_change'][scenario - 1].append(Elev_change[:, 2])
    scenario_data['percent_active_pulse'][scenario - 1].append(sorted_percent_active_pulse[:, -1])
    scenario_data['D_mean_pulse_each_link'][scenario - 1].append(sorted_D_mean_pulse_each_link[:, -1])
    scenario_data['volume_pulse_at_each_link'][scenario - 1].append(sorted_volume_pulse_at_each_link[:, -1])
    scenario_data['percent_pulse_remain'][scenario - 1].append(percent_pulse_remain)

    #XXX ######### Plots ###########


    # remain_vol_pulse = initial_vol_on_link_pulse - final_vol_on_link_pulse 
    # percent_remain_vol_pulse = (remain_vol_pulse/initial_vol_on_link_pulse)*100

    # # #Plot 1 v2

    # # plt.figure()
    # # plt.plot(elev_change_at_link)
    # # plt.title("Change in elevation through time (Scenario = %i)" %scenario)
    # # plt.ylabel("Elevation change (m)")
    # # plot_name = str(new_dir) + "/" + 'TrackElevChangev2(Scenario' + str(scenario_num) + ').png'
    # # plt.savefig(plot_name, dpi=700)
    # # #plt.savefig('TrackElevChangev2 (Scenario'+scenario_num+ ').png')

    # # #Plot 2
    # # # #### Tracking the output: are we at (or near) steady state?  ####
    # # fig, ax1 = plt.subplots()
    # # ax2 = ax1.twinx()

    # # ax1.plot(timestep_array, d_recycled_parcels,'-', color='brown')
    # # ax2.plot(timestep_array, n_recycled_parcels,'-', color='k') #,'.' makes it a dot chart, '-' makes line chart

    # # ax1.set_xlabel("Days")
    # # ax1.set_ylabel("Mean D recycled parcels (m)", color='k')
    # # plt.title("Recycled parcels (Scenario = %i)" %scenario)
    # # ax2.set_ylabel("Number of recycled parcels", color='brown')
    # # plot_name = str(new_dir) + "/" + 'Recycled(Scenario' + str(scenario_num) + ').png'
    # # plt.savefig(plot_name, dpi=700)
    # # #plt.savefig('Recycled (Scenario'+scenario_num+ ').png')
    # # plt.show()



    # #Plot 3
    # ####  How far is the pulse material transporting?    ####

    # # d_percent_loss = (parcels.dataset.D.values[mask_ispulse,-1] - new_D[:,0])/new_D[:,0]
    # # d_non_zero = parcels.dataset["D"].values[-num_pulse_parcels:,-1]
    # # d_non_zero[d_non_zero==0] = np.nan
    # # plt.scatter(d_non_zero,Pulse_cum_transport[:,-1],10,d_percent_loss, vmin=-0.1, vmax=-0.6)
    # # #plt.title("How far is the pulse material transporting? (Scenario = %i)" %scenario)
    # # plt.ylabel('Pulse parcel cumulative transport distance (m)')
    # # plt.xlabel('D (m)')
    # # plt.colorbar(label='Percentage of grains loss (%)')
    # # plt.ylim(0, 65900)
    # # plt.xlim(0, 0.04)
    # # # reference lines
    # # plt.axhline(y = link_len[0], color = 'grey', linestyle = ':') # plot a horizontal line
    # # plt.text(0.06,link_len[0]*1.1,'one link length')
    # # plt.axhline(y = np.sum(link_len), color = 'grey', linestyle = ':') # plot a horizontal line
    # # plt.text(0.06,link_len[160]*grid.number_of_links*0.95,'full channel length')
    # # #plt.savefig('DistTravelPulse(Scenario'+scenario_num+ ').png', dpi=700)
    # # plt.show()

    

    # #Plot 4
    # # How far have the non-pulse parcels traveled in total?
    # travel_dist = nst._distance_traveled_cumulative[:-total_num_pulse_parcels]
    # nonzero_travel_dist = travel_dist[travel_dist>0]
    # plt.hist(np.log10(nonzero_travel_dist),30) # better to make log-spaced bins...
    # plt.xlabel('log10( Cum. parcel travel distance (m) )')
    # plt.ylabel('Number of non-pulse parcels')
    # plt.title("How far have the non-pulse parcels traveled in total? (Scenario = %i)" %scenario)
    # #plt.savefig(plot_name, dpi=700)
    # plt.text(2,50, 'n = '+str(len(nonzero_travel_dist))+' parcels')
    # #plt.savefig('DistTravelNonPulse(Scenario'+scenario_num+ ').png')
    # plt.text(2,50, 'n = '+str(len(nonzero_travel_dist))+' parcels')
    
    # plt.figure()
    # plt.hist(np.log(SHRS_proxy), color = "blue", label= "SHRS Proxy")
    # plt.hist(np.log(Double_SHRS), color= "red", label= "Double SHRS Proxy")
    # plt.ylabel("Number of pulse parcels")
    # plt.xlabel("Log10 Abrasion rate (1/km)")
    # plt.legend()
    # # %%

    # #Plot 4
    # # What is the active parcel volume/total?
    # plt.figure()

    # # #plt.plot(grid.at_link["dist_upstream"],sediment_active_percent)
    # # plt.pcolor(time_array, dist_upstream, sed_total_vol)
    # # plt.title("Active parcel volume (Scenario = %i)" %scenario)
    # # plt.xlabel('Days')
    # # plt.ylabel('Distance from upstream (km)')
    # # #plt.savefig('percentActive (Scenario 1).png')
    # # plt.show()

    # plt.figure()
    # plt.pcolor(time_array, dist_upstream,sediment_active_percent)
    # plt.colorbar(label='average % active parcels')
    # plt.title("Active parcel volume (Scenario = %i)" %scenario)
    # plt.xlabel('Days')
    # plt.ylabel('Distance from upstream (km)')
    # plot_name = str(new_dir) + "/" + 'Percent_Active(Scenario' + str(scenario_num) + ').png'
    # plt.savefig(plot_name, dpi=700)
    # ##plt.savefig('percentActive (Scenario'+scenario_num+ ').png')
    # plt.show()

    # #Plot 5
    # #what is the number of parcels active per link
    # plt.figure()
    # plt.plot(sorted_num_active_parcels_each_link)
    # plt.title("Number of parcels active (Scenario = %i)" %scenario)
    # plt.xlabel("Link number downstream")
    # plt.ylabel("Number of active parcels")
    # plot_name = str(new_dir) + "/" + 'ActiveParcel(Scenario' + str(scenario_num) + ').png'
    # plt.savefig(plot_name, dpi=700)
    # #plt.savefig('ActiveParcel (Scenario'+scenario_num+ ').png')
    # plt.show()



    # # XXX
    # plt.figure()
    # plt.pcolor(time_array, dist_upstream, sorted_num_pulse_each_link)
    # plt.title("Number of total pulse parcels (Scenario = %i)" %scenario)
    # plt.colorbar(label='Num pulse parcel')
    # plt.xlabel("Days")
    # plt.ylabel("Distance from upstream (km)")
    # plot_name = str(new_dir) + "/" + 'TotalPulseParcel(Scenario' + str(scenario_num) + ').png'
    # plt.savefig(plot_name, dpi=700)
    # #plt.savefig('Total_pulseParcel (Scenario'+scenario_num+ ').png')
    # plt.show()

    # plt.figure()
    # plt.pcolor(time_array, dist_upstream, sorted_num_total_parcels_each_link)
    # plt.title("Number of total parcels (Scenario = %i)" %scenario)
    # plt.colorbar(label='Number of parcels')
    # plt.xlabel("Days")
    # plt.ylabel("Distance from upstream (km)")
    # plot_name = str(new_dir) + "/" + 'Total_parcel(Scenario' + str(scenario_num) + ').png'
    # plt.savefig(plot_name, dpi=700)
    # #plt.savefig('TotalParcel (Scenario'+scenario_num+ ').png')
    # plt.show()


    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()

    # ax1.plot(dist_upstream, D50_each_link,'-', color='k')
    # ax2.plot(dist_upstream, sorted_num_total_parcels_each_link,'-', color='brown') #,'.' makes it a dot chart, '-' makes line chart

    # ax1.set_xlabel("Distance from upstream (km)")
    # ax1.set_ylabel("D50", color='k')
    # plt.title("Number of parcels and the grain size (Scenario = %i)" %scenario)
    # ax2.set_ylabel("Number of total parcels", color='brown')
    # plot_name = str(new_dir) + "/" + 'RSM(Scenario' + str(scenario_num) + ').png'
    # plt.savefig(plot_name, dpi=700)
    # #plt.savefig('RSm (Scenario'+scenario_num+ ').png')
    # plt.show()

    # #%%
    # #Plot 6
    # #what is the  number of pulse parcels in the active layer

    # plt.figure()
    # num_active_pulse_nozero = sorted_num_active_pulse.copy()
    # num_active_pulse_nozero[num_active_pulse_nozero == 0]= np.nan


    # plt.pcolor(time_array, dist_upstream, num_active_pulse_nozero, cmap= 'winter_r')
    # #plt.title("How far is the pulse material transporting? (Scenario = %i)" %scenario)
    # plt.colorbar(label='Number of active pulse parcels')
    # plt.xlabel("Days")
    # plt.ylabel("Distance from upstream (km)")
    # plot_name = str(new_dir) + "/" + 'ActivePulse(Scenario' + str(scenario_num) + ').png'
    # plt.savefig(plot_name, dpi=700)
    # #plt.savefig('ActivePluse(Scenario'+scenario_num+ ').png', dpi=700)
    # plt.show()
    # #%%

    # # #Plot 7
    # # #what is d_mean_active downstream
    # # plt.figure()
    # # plt.pcolor(time_array, dist_upstream, active_d_mean)
    # # plt.title("What is the mean grain size through time? (Scenario = %i)" %scenario)

    # # plt.colorbar(label='Mean grain size (m)')
    # # plt.xlabel("Days")
    # # plt.ylabel("Link downstream")
    # # #plt.savefig('ActiveDmean(Scenario'+scenario_num+ ').png')
    # # plt.show()

    # #%% #plot8: Volume of pulse parcels through time 
    # plt.figure()
    # vol_pulse_nozero = sorted_volume_pulse_at_each_link.copy()
    # vol_pulse_nozero[vol_pulse_nozero == 0]= np.nan
    # plt.pcolor(time_array, dist_upstream,vol_pulse_nozero, cmap= 'plasma_r', norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=5000))
    # plt.colorbar(label='Volume of pulse parcels ($m^3$)')
    # #plt.title("Volume of pulse parcel through time (Scenario = %i)" %scenario)
    # plt.xlabel("Days")
    # plt.ylabel("Distance from upstream (km)")
    # plot_name = str(new_dir) + "/" + 'Vol_pulse(Scenario' + str(scenario_num) + ').png'
    # plt.savefig(plot_name, dpi=700)
    # plt.show()

    # #for all parcels
    # plt.figure()
    # vol_nozero = sorted_volume_at_each_link.copy()
    # vol_nozero[vol_nozero == 0]= np.nan
    # plt.pcolor(time_array, dist_upstream,vol_nozero, cmap= 'plasma_r', norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=5000))
    # plt.colorbar(label='Volume of all parcels ($m^3$)')
    # #plt.title("Volume of parcel through time (Scenario = %i)" %scenario)
    # plt.xlabel("Days")
    # plt.ylabel("Distance from upstream (km)")
    # plot_name = str(new_dir) + "/" + 'vol(Scenario' + str(scenario_num) + ').png'
    # plt.savefig(plot_name, dpi=700)
    # #plt.savefig('Vol_(Scenario'+scenario_num+ ').png', dpi=700)
    # plt.show()

    # #PLOT 9: D mean of pulse at each link
    # plt.figure()
    # D_mean_pulse_each_link_nozero= sorted_D_mean_pulse_each_link.copy()
    # D_mean_pulse_each_link_nozero[D_mean_pulse_each_link_nozero == 0]= np.nan
    # plt.pcolor(time_array, dist_upstream, D_mean_pulse_each_link_nozero, norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=1.0), cmap= 'winter_r')
    # cbar = plt.colorbar(label='Mean grain size of pulse parcels (m)')
    # # cbar.set_ticks([0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1])
    # # cbar.set_ticklabels(["0.01", "0.02", "0.03", "0.04", "0.06", "0.08", "0.1"])
    # #plt.title("Dmean grain size through time (Scenario = %i)" %scenario)
    # plt.xlabel("Days")
    # plt.ylabel("Distance from upstream (km)")
    # plot_name = str(new_dir) + "/" + 'D_mean_pulse(Scenario' + str(scenario_num) + ').png'
    # plt.savefig(plot_name, dpi=700)
    # plt.show()

    # #for all parcels
    # plt.figure()
    # D_mean_each_link_nozero= sorted_D_mean_each_link.copy()
    # D_mean_each_link_nozero[D_mean_each_link_nozero == 0]= np.nan
    # plt.pcolor(time_array, dist_upstream, D_mean_each_link_nozero, norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=0.1), cmap= 'winter_r')
    # cbar = plt.colorbar(label='Mean grain size of all parcels (m)')
    # # cbar.set_ticks([0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1])
    # # cbar.set_ticklabels(["0.01", "0.02", "0.03", "0.04", "0.06", "0.08", "0.1"])
    # #plt.title("Dmean grain size through time (Scenario = %i)" %scenario)
    # plt.xlabel("Days")
    # plt.ylabel("Distance from upstream (km)")
    # plot_name = str(new_dir) + "/" + 'Dmean_(Scenario' + str(scenario_num) + ').png'
    # plt.savefig(plot_name, dpi=700)
    # #plt.savefig('Dmean_(Scenario'+scenario_num+ ').png', dpi=700)
    # plt.show()

    # #PLOT 10: percentage active parcels
    # plt.figure()
    # plt.pcolor(time_array, dist_upstream, sorted_percent_active_pulse, cmap= 'Wistia', norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=1.0))
    # cbar = plt.colorbar(label='Percentage of active pulse parcels (%)')
    # cbar.set_ticks([0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
    # cbar.set_ticklabels(["0.1", "0.2", "0.3", "0.4", "0.6", "0.8", "1.0"])
    # #plt.title("Percentage active pulse parcels (Scenario = %i)" %scenario)
    # plt.xlabel("Days")
    # plt.ylabel("Distance from upstream (km)")
    # plot_name = str(new_dir) + "/" + 'percent_active_pulse(Scenario' + str(scenario_num) + ').png'
    # plt.savefig(plot_name, dpi=700)
    # plt.show()

    # # if scenario == 1:
    # #     def scenario1_percent_active_pulse ()
    # #for all parcels
    # plt.figure()
    # percent_active= num_active/ num_each_link
    # sorted_percent_active = percent_active[index_sorted_area_link] 
    # sorted_percent_active[sorted_percent_active == 0]= np.nan
    # plt.pcolor(time_array, dist_upstream, sorted_percent_active, cmap= 'Wistia', norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=1.0))
    # cbar = plt.colorbar(label='Percentage of active parcels (%)')
    # cbar.set_ticks([0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
    # cbar.set_ticklabels(["0.1", "0.2", "0.3", "0.4", "0.6", "0.8", "1.0"])
    # #plt.title("Percentage active parcels (Scenario = %i)" %scenario)
    # plt.xlabel("Days")
    # plt.ylabel("Distance from upstream (km)")
    # plot_name = str(new_dir) + "/" + 'percent_active(Scenario' + str(scenario_num) + ').png'
    # plt.savefig(plot_name, dpi=700)
    # #plt.savefig('percent_active_(Scenario'+scenario_num+ ').png', dpi=700)
    # plt.show()
    # #%%

    # # PLOT 1 v3: elevation change through time, as pcolor
    # plt.figure(" Track elev change")
    # plt.pcolor(time_array, dist_upstream_nodes, Elev_change, shading='auto', norm=colors.CenteredNorm(), cmap='coolwarm') 
    # plt.colorbar(label='Elevation change from initial (m)')
    # #plt.title("Elevation change through time (Scenario = %i)" %scenario)
    # plt.xlabel('Days')
    # plt.ylabel('Distance from upstream (km)')
    # plot_name = str(new_dir) + "/" + 'TrackElevChange(Scenario' + str(scenario_num) + ').png'
    # plt.savefig(plot_name, dpi=700)
    # plt.show()

    # plt.figure("Transport Capacity")
    # plt.pcolor(time_array, dist_upstream, jon_W)
    # cbar = plt.colorbar(label='Transport rate (m/s)')
    # plt.xlabel("Days")
    # plt.ylabel("Distance from upstream (km)")
    # plot_name = str(new_dir) + "/" + 'transport_capacity(Scenario' + str(scenario_num) + ').png'
    # plt.savefig(plot_name, dpi=700)
    # plt.show()


    # half_width = width/2
    # log_width = np.log(width)
    # fig = plot_network_and_parcels(grid,parcels,parcel_time_index=0,link_attribute=('elevation_change'),parcel_alpha=0,network_linewidth=log_width, link_attribute_title= "Elevation Change in each reach (m)", network_cmap= "coolwarm")
    # plot_name = str(new_dir) + "/" + 'NST_ELEVCHANGE(Scenario' + str(scenario_num) + ').jpg'
    # fig.savefig(plot_name, bbox_inches='tight', dpi=700)
    # fig =plot_network_and_parcels(grid,parcels,parcel_time_index=0, link_attribute=('change_D'),parcel_alpha=0, link_attribute_title="Diameter [m]", network_cmap= "coolwarm")
    # plot_name = str(new_dir) + "/" + 'NST_grain_size_change(Scenario' + str(scenario_num) + ').jpg'
    # fig.savefig(plot_name, bbox_inches='tight', dpi=700)
    
    scenario_labels = {
    0: 'No abrasion',
    1: 'Distribution of abrasion rates',
    2: 'Double SHRS proxy abrasion'}
    
    for variable, data in scenario_data.items():
        plt.figure()
        plt.title(variable)
        for scen, scenario_values in enumerate(data):
            for i, array in enumerate(scenario_values):
                if variable == 'elevation_change':
                    plt.plot(dist_upstream_nodes, array, label= f'{scenario_labels[scen]}', color=['green', 'blue', 'red'][scen])
                    plt.xlabel('Distance from upstream (km)')
                elif variable == 'percent_pulse_remain':
                   plt.plot(time_array, array, label= f'{scenario_labels[scen]}', color=['green', 'blue', 'red'][scen]) 
                   plt.xlabel('Days')
                else:
                    plt.plot(dist_upstream, array, label= f'{scenario_labels[scen]}', color=['green', 'blue', 'red'][scen])
                    plt.xlabel('Distance from upstream (km)')
        plt.ylabel(variable.replace('_', ' ').title())
        #plt.title(f'{variable.replace("_", " ").title()} for Different Scenarios')
        plt.legend(loc="upper right")
        plot_name = str(new_dir) + "/" + variable + str(scenario_num) + ').png'
        plt.savefig(plot_name, dpi=700)
        
    plt.show()
    return array
# for link in range(number_of_links):
#         # Find parcels associated with the current link
#     link_parcel_indices = np.where(element_all_parcels[mask_active] == link)[0]
        
#         # If there are parcels associated with the current link
#     if len(link_parcel_indices) > 0:
#             # Get values of parcels associated with the current link
#         parcel_values = grain_sizes[link_parcel_indices]
            
#             # Sort the parcel values from smallest to largest
#         sorted_values = np.sort(parcel_values)
#     D16= sorted_values[int(np.size(sorted_values)*0.16)] 

# plt.figure()
# plt.hist(np.log10(SHRS_proxy), color = "blue", label= "SHRS Proxy")
# plt.hist(np.log10(Double_SHRS), color= "red", alpha= 0.7, label= "Double SHRS Proxy")
# plt.ylabel("Number of pulse parcels")
# plt.xlabel("Log10 Abrasion rate (1/m)")
# plt.legend()  

# plt.figure()
# plt.hist(new_density)
# plt.ylabel("Number of pulse parcels")
# plt.xlabel("Density (kg/$m^3$)")
    
def main():
    for scenario_ in range (1,4):
        run_scenario(scenario_)
        # Plot all scenarios for each variable on separate plots
    # Call run_scenario outside the loop
    scenario_labels = {
    0: 'No abrasion',
    1: 'SHRS proxy abrasion',
    2: 'Double SHRS proxy abrasion'}
    
    # for variable, data in scenario_data.items():
    #     plt.figure()
    #     plt.title(variable)
    #     for scen, scenario_values in enumerate(data):
    #         for i, array in enumerate(scenario_values):
    #             if variable == 'elevation_change':
    #                 plt.plot(dist_upstream_nodes, array, label= f'{scenario_labels[scen]}', color=['green', 'blue', 'red'][scen])
    #                 plt.xlabel('Distance from upstream (km)')
    #             elif variable == 'percent_pulse_remain':
    #                plt.plot(time_array, array, label= f'{scenario_labels[scen]}', color=['green', 'blue', 'red'][scen]) 
    #                plt.xlabel('Days')
    #             else:
    #                 plt.plot(dist_upstream, array, label= f'{scenario_labels[scen]}', color=['green', 'blue', 'red'][scen])
    #                 plt.xlabel('Distance from upstream (km)')
    #     plt.ylabel(variable.replace('_', ' ').title())
    #     #plt.title(f'{variable.replace("_", " ").title()} for Different Scenarios')
    #     plt.legend(loc="upper right")
        
    # plt.show()

        
if __name__=="__main__":
    main()


