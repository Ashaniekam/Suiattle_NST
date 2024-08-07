# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:25:51 2024

@author: longrea
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is the code for my thesis on the Suiattle River


Tasks 6-18-24:
    - Sorting!! eek. Any variable that is sorted needs to have _DS as the suffix
   




@author: longrea, pfeiffea

Goal: Sort out new Suiattle shapefile issues

"""


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
from landlab.data_record.aggregators import aggregate_items_as_mean
from landlab.data_record.aggregators import aggregate_items_as_sum
from landlab.data_record.aggregators import aggregate_items_as_count



OUT_OF_NETWORK = NetworkModelGrid.BAD_INDEX - 1

# #### Selecting abrasion/density scenario #####
scenario = 2

if scenario == 1:
    scenario_num = "none"
    model_state = "Abrasion run_no_abrasion"
elif scenario == 2:
    scenario_num = "SHRS proxy"
    model_state = "Abrasion run_SHRS proxy"
else: 
    scenario_num = "testing_pulse_transport"
    model_state = "Test"

# ##### Basic model parameters 

timesteps = 5
pulse_time = 2
dt = 60*60*24*1  # len of timesteps (1 day)

# bookkeeping 
new_dir_name = "Results and plots_" + model_state
new_dir = pathlib.Path(os.getcwd(), new_dir_name)
new_dir.mkdir(parents=True, exist_ok=True)



# %% ##### Set up the grid, channel characteristics #####
shp_file = os.path.join(os.getcwd (), ("Suiattle_river.shp"))
points_shapefile = os.path.join(os.getcwd(), ("Suiattle_nodes.shp"))

#Suiattle_Dem = np.loadtxt("10m_hillshade_suia.asc", skiprows=6)


grid = read_shapefile(
    shp_file,
    points_shapefile = points_shapefile,
    node_fields = ["usarea_km2", "elev_m"],
    link_fields = ["usarea_km2", "Length_m", "Slope", "Width_m"],
    link_field_conversion = {"usarea_km2": "drainage_area", "Slope":"channel_slope", "Width_m": "channel_width", "Length_m":"reach_length", }, #, "Width": "channel_width";  how to read in channel width
    node_field_conversion = {"usarea_km2": "drainage_area", "elev_m": "topographic__elevation"},
    threshold = 0.1,
    )

q_dataframe = pd.read_excel((os.path.join(os.getcwd (), ("Discharge from DHSVM Suiattle.xlsx"))), sheet_name = 'Discharge (2% exceedance)', header = 0)
discharge = q_dataframe['Q'].values

#########
# Landlab orders links and nodes are numbered from zero in the bottom left corner of the grid, 
# then run along each row in turn. This numnbering technique does not number reaches/links in upstream/downstream order. To
# get the network in upstream order, the links are sorted based on drainage area. All sorted variables end with "_DS"

#Using drainage area for links and nodes to create ordering indices
area_link = grid.at_link["drainage_area"]
area_node = grid.at_node["drainage_area"]

index_sorted_area_link = (area_link).argsort()
index_sorted_area_node = (area_node).argsort()

#Sorted Drainage Area
area_link_DS = area_link[index_sorted_area_link]
area_node_DS = area_node[index_sorted_area_node]

#Sorted Topographic Elevation and Bedrock Elevation
topo = grid.at_node['topographic__elevation'].copy()
topo[topo== 776.32598877] += 3 #attemp to smooth the grid to get rid of bottleneck
topo = topo[index_sorted_area_node]
grid.at_node["bedrock__elevation"] = topo.copy()

#Sorted Width
width = grid.at_link["channel_width"]
width_DS = width[index_sorted_area_link]

#Sorted Slope
slope = grid.at_link["channel_slope"]
slope_DS = slope[index_sorted_area_link]
initial_slope= slope.copy()

#Sorted Length
length = grid.at_link["reach_length"]
length_DS = length[index_sorted_area_link]

grid.at_link["dist_upstream"]= np.cumsum(length_DS)
dist_upstream = grid.at_link["dist_upstream"]/1000
dist_upstream_nodes = np.insert(dist_upstream, 0, 0.0)

#to calculate flow depth
Mannings_n = 0.086 #median calculated from Suiattle gages (units m^3/s)
grid.at_link["flow_depth"] = ((Mannings_n*discharge)/ ((slope**0.5)*width))**0.6

depth = grid.at_link["flow_depth"].copy()

# %% Set up parcels 
rho_water = 1000
rho_sed = 2650
gravity = 9.81
tau = rho_water * gravity * depth * slope

number_of_links = grid.number_of_links

#creating sediment parcels in the DATARECORD

tau_c_multiplier = 2.4

initialize_parcels = BedParcelInitializerDepth(
    grid,
    flow_depth_at_link = depth,
    tau_c_50 = 0.15* slope**0.25, # slope dependent critical shields stress
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


# calculation of the initial volume of sediment on each link

initial_vol_on_link = np.empty(number_of_links, dtype = float)
initial_vol_on_link= aggregate_items_as_sum(
    parcels.dataset.element_id.values[:, -1].astype(int),
    parcels.dataset.volume.values[:, -1],
    number_of_links,
)

D50_each_link = parcels.calc_aggregate_value(
    xr.Dataset.median, "D", at="link", fill_value=0.0)


# %% XX Plots related to grid... 

# Longitudinal Profile
plt.figure()
plt.plot(dist_upstream, grid.at_link["channel_slope"], '-' )
plt.title("Starting channel Slope")
plt.ylabel("Slope")
plt.xlabel("Distance downstream (km)")
plt.show()

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

tau_star = tau/((rho_sed - rho_water) * gravity * D50_each_link)

# Plot for shield stress
plt.figure()
plt.title("Tau_star")
plt.plot(dist_upstream, tau_star)
plt.ylabel("Shield Stress")
plt.xlabel("Distance upstream (km)")
plt.show()

# D50 for each link
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


# %% Instantiate NST

# flow direction
fd = FlowDirectorSteepest(grid, "topographic__elevation")
fd.run_one_step()
bed_porosity=0.3

# initialize the networksedimentTransporter
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
    k_transp_dep_abr= 15.0,
)

# %% Setup for plotting during/after loop

#variables needed for plots
Pulse_cum_transport = []

# Tracked for each timestep
n_recycled_parcels = np.ones(timesteps)*np.nan
vol_pulse_left = np.ones(timesteps)*np.nan
vol_pulse_on = np.ones(timesteps)*np.nan
d_recycled_parcels = np.ones(timesteps)*np.nan
timestep_array = np.arange(timesteps)

# Tracked for each link, for each timestep
sed_total_vol = np.ones([grid.number_of_links,timesteps])*np.nan
sediment_active_percent = np.ones([grid.number_of_links,timesteps])*np.nan
active_d_mean= np.ones([timesteps,grid.number_of_links])*np.nan
D_mean_pulse_each_link= np.ones([grid.number_of_links,timesteps])*np.nan
D_mean_each_link= np.ones([grid.number_of_links,timesteps])*np.nan
num_pulse_each_link= np.ones([grid.number_of_links,timesteps], dtype=int)*np.nan
num_each_link= np.ones([grid.number_of_links,timesteps], dtype=int)*np.nan
num_total_parcels_each_link= np.ones([grid.number_of_links,timesteps], dtype=int)*np.nan
transport_capacity= np.ones([grid.number_of_links,timesteps])*np.nan

num_active_parcels_each_link = np.ones([grid.number_of_links,timesteps])*np.nan
volume_at_each_link = np.ones([grid.number_of_links,timesteps])*np.nan
volume_pulse_at_each_link = np.ones([grid.number_of_links,timesteps])*np.nan
num_active_pulse= np.ones([grid.number_of_links,timesteps])*np.nan

time_array= np.arange(1, timesteps + 1)
time_array_ss= np.arange(pulse_time, timesteps+1)#ss = steady state
canyon_reaches= np.arange(45, 55)

Elev_change = np.empty([grid.number_of_nodes,timesteps]) # 2d array of elevation change in each timestep 


# %% Model runs

for t in range(0, (timesteps*dt), dt):
    #by index of original sediment
    start_time = model_time.time()
    
    
    # Masking parcels
    #boolean mask of parcels that have left the network
    mask1 = parcels.dataset.element_id.values[:,-1] == OUT_OF_NETWORK
    #boolean mask of parcels that are initial bed parcels (no pulse parcels)
    mask2 = parcels.dataset.source == "initial_bed_sed" 
    
    bed_sed_out_network = parcels.dataset.element_id.values[mask1 & mask2, -1] == -2
    
    
    
    
    # ###########   Parcel recycling by drainage area    ###########
    
    #index of the bed parcels that have left the network
    index_exiting_parcels = np.where(bed_sed_out_network == True) 
    
    #number of parcels that will be recycled
    num_exiting_parcels = np.size(index_exiting_parcels)

    # Calculate the total sum of area_link
    total_area = sum(area_link)

    # Calculate the proportions based on drainage area
    proportions = area_link/total_area

    # Calculate the number of parcels for each link based on proportions
    num_recycle_bed = [int(proportion * num_exiting_parcels) for proportion in proportions]

    # Calculate the remaining parcels not accounted for by using int above
    remaining_bed_sed = num_exiting_parcels - sum(num_recycle_bed)

    # Calculate the number of additional parcels to add to each proportion
    additional_bed_sed = np.round(np.array(proportions) * remaining_bed_sed).astype(int)

    # Distribute the remaining parcels to the proportions with the largest values
    while remaining_bed_sed > 0:
        max_index = np.argmax(additional_bed_sed)
        additional_bed_sed[max_index] -= 1
        num_recycle_bed[max_index] += 1
        remaining_bed_sed -= 1


    indices_recyc_bed = []

    # Generate indices for each proportion
    for i, num in enumerate(num_recycle_bed):
        indices_recyc_bed.extend([i] * num)

    indices_recyc_bed = np.squeeze(indices_recyc_bed)

    # assign the new starting link
    for bed_sed in index_exiting_parcels:
        parcels.dataset.element_id.values[bed_sed, 0] = indices_recyc_bed
        recyc_= np.where(parcels.dataset.element_id.values[bed_sed,0] == indices_recyc_bed)
        
        for recycle_destination in indices_recyc_bed:
            parcels.dataset["recycle_destination"].values[recyc_] = np.full(num_exiting_parcels, "recycle_des %s" %recycle_destination)

    n_recycled_parcels[np.int64(t/dt)]=np.sum(parcels.dataset.element_id.values[:,-1] == OUT_OF_NETWORK,-1)
    d_recycled_parcels[np.int64(t/dt)]=np.mean(parcels.dataset.D.values[parcels.dataset.element_id.values[:,-1] == OUT_OF_NETWORK,-1])
    

    # #######   Ta da!    #########     
      
    nst.run_one_step(dt)
    
    # #### another day passed... #####
    
    
                ########## Making a pulse #############
    if t==dt*pulse_time: 
        print('making a pulse')
        print('t = ',t, ',timestep = ', t/dt)
        
        
        fan_thickness = np.array([6.25, 6.25, 5.75, 2.75]) #meters
        fan_location= np.array([7, 8, 9, 10])
        pulse_parcel_vol = 10 #volume of each parcel m^3
        pulse_volume = fan_thickness*length[fan_location]*width[fan_location] # m^3; volume of pulse that should be added to each link of the Chocolate Fan
        pulse_rock_volume = pulse_volume * (1-bed_porosity)
        num_pulse_parcels_by_vol = pulse_rock_volume/pulse_parcel_vol #number of parcels added to each link based on volume; 
        total_num_pulse_parcels = int(np.sum(num_pulse_parcels_by_vol))
        pulse_location = [index for index, freq in enumerate((num_pulse_parcels_by_vol).astype(int), start=7) for _ in range(freq)]
        random.shuffle(pulse_location)
        
        
        
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

        # %% Abrasion scenarios using Allison's field data
        field_data = pd.read_excel((os.path.join(os.getcwd (), ("SuiattleFieldData_Combined20182019.xlsx"))), sheet_name = '4-SHRSdfDeposit', header = 0)
        SHRS = field_data['SHRS median'].values
        SHRS_MEAN = np.mean(SHRS)
        SHRS_STDEV = np.std(SHRS)

        SHRS_distribution = np.random.normal(SHRS_MEAN, SHRS_STDEV, (np.size(newpar_element_id)))


        measured_alphas = 3.0715*np.exp(-0.136*SHRS_distribution) # units: 1/km
         

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
            new_abrasion_rate = (tumbler_4/1000)* np.ones(np.size(newpar_element_id)) #tumbler correction 4 !!!!
            new_density = 894.978992640976*np.log(SHRS_distribution)-1156.7599235065895
        else: 
            new_abrasion_rate = (tumbler_4/1000) * np.ones(np.size(newpar_element_id))
         
        
        new_D= np.random.lognormal(np.log(0.09),np.log(1.5),np.shape(newpar_element_id))    # (m) the diameter of grains in each pulse parcel 
        
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
        
#%% #Calculate Variables and tracking timesteps
    if t == parcels.dataset.time[0].values: # if we're at the first timestep

        avg_init_sed_thickness = np.mean(
            grid.at_node['topographic__elevation'][:-1]-grid.at_node['bedrock__elevation'][:-1])        
        grid.at_node['bedrock__elevation'][-1]=grid.at_node['bedrock__elevation'][-1]+avg_init_sed_thickness


        elev_initial = grid.at_node['topographic__elevation'].copy()
    Elev_change[:,np.int64(t/dt)] = grid.at_node['topographic__elevation']-elev_initial

    #Tracking timesteps
    print("Model time: ", t/(60*60*24), "days passed (", t/dt, 'timesteps)')
    print('Elapsed:', (model_time.time() - start_time)/60 ,' minutes')
    
    # Print some pertinent information about the model run
    print('mean grain size of pulse',np.mean(
        parcels.dataset.D.values[parcels.dataset.source.values[:] == 'pulse',-1]))
    
# %% Calculating transport capacity and volume remaining after abrasion
    #Additional masks
    #boolean mask of parcels that are pulse parcels
    mask_ispulse = parcels.dataset.source == 'pulse'
    #boolean mask of parcels that are active
    mask_active = parcels.dataset.active_layer[:,-1]==1 
    #boolean mask of element IDs for all parcels
    element_all_parcels = parcels.dataset.element_id.values[:, -1].astype(int)
    #boolean mask of element IDs that are pulse parcels
    element_pulse= element_all_parcels[mask_ispulse]
    #boolean mask of pulse parcels' volumes
    volume_of_pulse = parcels.dataset.volume.values[mask_ispulse,-1]
    
    
    #Calculating transport capacity
    active_parcels = parcels.dataset.element_id.values[mask_active,-1].astype(int) #array of active parcels' element IDs
    parcel_volume = parcels.dataset.volume.values[:, -1] #parcel volume
    active_layer_volume = nst._active_layer_thickness *width*length
    
    weighted_sum_velocity = 0
    for parcel_id in active_parcels:
        parcel_velocity = nst._pvelocity[parcel_id]
        parcel_vol = parcel_volume[parcel_id]
        weighted_sum_velocity += parcel_velocity * parcel_vol

    weighted_mean_velocity = weighted_sum_velocity / active_layer_volume #divide that by active layer volume 
    # Convert weighted-mean velocity to m/s
    transport_capacity [:,np.int64(t/dt)]= weighted_mean_velocity * nst._active_layer_thickness * width  #multiply by active layer thickness and link width 
   
    
    # Calculating volume of pulse that remains after abrasion
    volume_of_pulse = parcels.dataset.volume.values[mask_ispulse,-1]

    vol_pulse_on[np.int64(t/dt)]=np.sum(volume_of_pulse[element_pulse!= OUT_OF_NETWORK])
    vol_pulse_left[np.int64(t/dt)]=np.sum(volume_of_pulse[element_pulse == OUT_OF_NETWORK])
     
    percent_pulse_remain = (vol_pulse_on/(vol_pulse_on + vol_pulse_left))*100
     
    
#%%  #Aggregators for all parcels
    
    # Populating arrays with data from this timestep. 
    sed_act_vol= grid.at_link["sediment__active__volume"][index_sorted_area_link] 
  
    sed_total_vol[:,np.int64(t/dt)] = grid.at_link["sediment_total_volume"][index_sorted_area_link]
    sediment_active_percent[:,np.int64(t/dt)]= sed_act_vol/sed_total_vol[:,np.int64(t/dt)]
    
    num_active_parcels_each_link [:,np.int64(t/dt)]= aggregate_items_as_sum(
        element_all_parcels,
        parcels.dataset.active_layer.values[:, -1],
        number_of_links,
    )
    
    volume_at_each_link[:,np.int64(t/dt)]= aggregate_items_as_sum(
        element_all_parcels,
        parcels.dataset.volume.values[:, -1],
        number_of_links, 
    )
    
    D_mean_each_link[:,np.int64(t/dt)] = aggregate_items_as_mean(
        element_all_parcels,
        parcels.dataset.D.values[:,-1],
        parcels.dataset.volume.values[:, -1],
        number_of_links,
    )
    
    num_total_parcels_each_link[:, np.int64(t/dt)] = aggregate_items_as_count(
        element_all_parcels,
        number_of_links,
        )
    
    num_each_link[:, np.int64(t/dt)] = aggregate_items_as_count(
        element_all_parcels,
        number_of_links,
        )
    
#%% #Aggregators for pulse parcel calcualtions/queries
    
    if t > dt*pulse_time :
        Pulse_cum_transport = np.append(
                 Pulse_cum_transport,
                 np.expand_dims(
                         nst._distance_traveled_cumulative[-total_num_pulse_parcels:],
                         axis = 1),
                 axis = 1
                 )
        num_active_pulse [:,np.int64(t/dt)]= aggregate_items_as_sum(
              element_pulse,
              parcels.dataset.active_layer.values[mask_ispulse, -1],
              number_of_links,
              )
    
        volume_pulse_at_each_link [:,np.int64(t/dt)]= aggregate_items_as_sum(
            element_pulse,
            parcels.dataset.volume.values[mask_ispulse, -1],
            number_of_links,
            )
 
        num_pulse_each_link[:, np.int64(t/dt)] = aggregate_items_as_count(
            element_pulse,
            number_of_links,
            )
        
        D_mean_pulse_each_link[:,np.int64(t/dt)]= aggregate_items_as_mean(
            element_pulse,
            parcels.dataset.D.values[mask_ispulse,-1],
            parcels.dataset.volume.values[mask_ispulse, -1],
            number_of_links,
            )
        

    # D_MEAN_ACTIVE through time
    active_d_mean [np.int64(t/dt),:]= nst.d_mean_active
    initial_d = active_d_mean[0]

    thickness_at_link = volume_pulse_at_each_link[:,2]/grid.at_link["channel_width"]/grid.at_link["reach_length"]/(1-bed_porosity)

#%% Sorted Variables
num_active_parcels_each_link_DS = num_active_parcels_each_link[index_sorted_area_link]
volume_at_each_link_DS = volume_at_each_link[index_sorted_area_link]
D_mean_each_link_DS = D_mean_each_link[index_sorted_area_link]
num_total_parcels_each_link_DS = num_total_parcels_each_link[index_sorted_area_link]
num_each_link_DS =  num_each_link[index_sorted_area_link]
num_active_pulse_DS = num_active_pulse[index_sorted_area_link] 
volume_pulse_at_each_link_DS = volume_pulse_at_each_link[index_sorted_area_link]
num_pulse_each_link_DS =  num_pulse_each_link[index_sorted_area_link]
D_mean_pulse_each_link_DS = D_mean_pulse_each_link[index_sorted_area_link]

num_active_pulse_nozero_DS = num_active_pulse_DS.copy()
num_active_pulse_nozero_DS[num_active_pulse_nozero_DS == 0]= np.nan

vol_pulse_nozero_DS = volume_pulse_at_each_link_DS.copy()
vol_pulse_nozero_DS[vol_pulse_nozero_DS == 0]= np.nan

vol_nozero_DS = volume_at_each_link_DS.copy()
vol_nozero_DS[vol_nozero_DS == 0]= np.nan

D_mean_pulse_each_link_nozero_DS= D_mean_pulse_each_link_DS.copy()
D_mean_pulse_each_link_nozero_DS[D_mean_pulse_each_link_nozero_DS == 0]= np.nan

D_mean_each_link_nozero_DS= D_mean_each_link_DS.copy()
D_mean_each_link_nozero_DS[D_mean_each_link_nozero_DS == 0]= np.nan

percent_active_pulse= num_active_pulse/ num_pulse_each_link
percent_active_pulse_DS = percent_active_pulse[index_sorted_area_link] 
percent_active_pulse_DS[percent_active_pulse_DS == 0]= np.nan

percent_active= num_active_parcels_each_link/ num_each_link
percent_active_DS = percent_active[index_sorted_area_link] 
percent_active_DS[percent_active_DS == 0]= np.nan
#%% Editing and saving variables in DataRecord and text file
parcels_dataframe= parcels.dataset.to_dataframe()
parcels_dataframe.to_csv('Parcels_data_scenario.csv', sep=',', index=True) 

#Saving a text file of model characteristics
timestep_in_days = dt/86400
# Converting the time in seconds to a timestamp
c_ti = model_time.ctime(os.path.getctime("NST_Suiattle_single_scenario_na_July_19.py"))
text_file = os.path.join(new_dir, 'TESTING ('+scenario_num+ ').txt')   
file = open('Suiattle_run_characteristics'+scenario_num+ ').txt', 'w')
model_characteristics = ["This script was created at %s" %c_ti, "This code is running for scenario %s" %scenario,  "This is the tau_c_multiplier %s" %tau_c_multiplier, 
                          "This is the number of timesteps %s" %timesteps, "The number of days in each timestep is %s" % timestep_in_days, # length of dt in days
                          "The pulse is added at timestep %s" %pulse_time,
                          "The volume of each pulse parcel is %s" %pulse_parcel_vol, "This is the number of pulse parcels added to the network %s"
                          %total_num_pulse_parcels,
                          
                          "The current stage of this code is editing variables to be normalized since steady state."]
for line in model_characteristics:
    file.write(line)
    file.write('\n')
file.close()

# %% ####   Plotting and analysis of results    ############  

# final volume of sediment on each link
final_vol_on_link = np.empty(number_of_links, dtype = float)
final_vol_on_link= aggregate_items_as_sum(
    element_all_parcels,
    parcels.dataset.volume.values[:, -1],
    number_of_links,
)
sorted_final_vol_on_link = final_vol_on_link[index_sorted_area_link] 


## Adding elevation change as an at_link variable
elev_change_at_link = ((final_vol_on_link - initial_vol_on_link) /
                        (grid.at_link["reach_length"]*grid.at_link["channel_width"]))  # volume change in link/ link bed area

# scenario_data['elevation_change'][scenario - 1].append(Elev_change[:, 2])
# scenario_data['percent_active_pulse'][scenario - 1].append(percent_active_pulse_DS[:, -1])
# scenario_data['D_mean_pulse_each_link'][scenario - 1].append(D_mean_pulse_each_link_DS[:, -1])
# scenario_data['volume_pulse_at_each_link'][scenario - 1].append(volume_pulse_at_each_link_DS[:, -1])
# scenario_data['percent_pulse_remain'][scenario - 1].append(percent_pulse_remain)

# ######### Plots ###########

#Elevation change
plt.figure()
plt.plot(elev_change_at_link)
plt.title("Change in elevation through time (Scenario = %i)" %scenario)
plt.ylabel("Elevation change (m)")
plot_name = str(new_dir) + "/" + 'TrackElevChangev2(Scenario' + str(scenario_num) + ').png'
plt.savefig(plot_name, dpi=700)
#plt.savefig('TrackElevChangev2 (Scenario'+scenario_num+ ').png')

# Recycled Parcels
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(timestep_array, d_recycled_parcels,'-', color='brown')
ax2.plot(timestep_array, n_recycled_parcels,'-', color='k') #,'.' makes it a dot chart, '-' makes line chart
ax1.set_xlabel("Days")
ax1.set_ylabel("Mean D recycled parcels (m)", color='k')
plt.title("Recycled parcels (Scenario = %i)" %scenario)
ax2.set_ylabel("Number of recycled parcels", color='brown')
plot_name = str(new_dir) + "/" + 'Recycled(Scenario' + str(scenario_num) + ').png'
plt.savefig(plot_name, dpi=700)
plt.show()

# How far have the non-pulse parcels traveled in total?
travel_dist = nst._distance_traveled_cumulative[:-total_num_pulse_parcels]
nonzero_travel_dist = travel_dist[travel_dist>0]
plt.hist(np.log10(nonzero_travel_dist),30) # better to make log-spaced bins...
plt.xlabel('log10( Cum. parcel travel distance (m) )')
plt.ylabel('Number of non-pulse parcels')
plt.title("How far have the non-pulse parcels traveled in total? (Scenario = %i)" %scenario)
plt.savefig(plot_name, dpi=700)
plt.text(2,50, 'n = '+str(len(nonzero_travel_dist))+' parcels')
plt.text(2,50, 'n = '+str(len(nonzero_travel_dist))+' parcels')



variables = {
    'sediment_active_percent': sediment_active_percent,
    'num_pulse_each_link_DS': num_pulse_each_link_DS,
    'num_total_parcels_each_link_DS': num_total_parcels_each_link_DS,
    'num_active_pulse_nozero_DS': num_active_pulse_nozero_DS,
    'vol_pulse_nozero_DS': vol_pulse_nozero_DS,
    'vol_nozero_DS': vol_nozero_DS,
    'D_mean_pulse_each_link_nozero_DS': D_mean_pulse_each_link_nozero_DS,
    'D_mean_each_link_nozero_DS': D_mean_each_link_nozero_DS,
    'percent_active_pulse_DS': percent_active_pulse_DS,
    'percent_active_DS': percent_active_DS,
    'Elev_change': Elev_change,
    'transport_capacity': transport_capacity,
}

# Define colorbar labels with new names
colorbar_labels = {
    'sediment_active_percent': 'average % active parcels',
    'num_pulse_each_link_DS': 'Num pulse parcel',
    'num_total_parcels_each_link_DS': 'Number of parcels',
    'num_active_pulse_nozero_DS': 'Number of active pulse parcels',
    'vol_pulse_nozero_DS': 'Volume of pulse parcels ($m^3$)',
    'vol_nozero_DS': 'Volume of all parcels ($m^3$)',
    'D_mean_pulse_each_link_nozero_DS': 'Mean grain size of pulse parcels (m)',
    'D_mean_each_link_nozero_DS': 'Mean grain size of all parcels (m)',
    'percent_active_pulse_DS': 'Percentage of active pulse parcels (%)',
    'percent_active_DS': 'Percentage of active parcels (%)',
    'Elev_change': 'Elevation change from initial (m)',
    'transport_capacity': 'Transport rate (m/s)',
}
# Define plot titles with new names
plot_titles = {
    'sediment_active_percent': f"Active parcel volume (Scenario = {scenario})",
    'num_pulse_each_link_DS': f"Number of total pulse parcels (Scenario = {scenario})",
    'num_total_parcels_each_link_DS': f"Number of total parcels (Scenario = {scenario})",
    'num_active_pulse_nozero_DS': f"How far is the pulse material transporting? (Scenario = {scenario})",
    'vol_pulse_nozero_DS': f"Volume of pulse parcel through time (Scenario = {scenario})",
    'vol_nozero_DS': f"Volume of parcel through time (Scenario = {scenario})",
    'D_mean_pulse_each_link_nozero_DS': f"Dmean grain size through time (Scenario = {scenario})",
    'D_mean_each_link_nozero_DS': f"Dmean grain size through time (Scenario = {scenario})",
    'percent_active_pulse_DS': f"Percentage active pulse parcels (Scenario = {scenario})",
    'percent_active_DS': f"Percentage active parcels (Scenario = {scenario})",
    'Elev_change': f"Elevation change through time (Scenario = {scenario})",
    'transport_capacity': f"Transport capacity (Scenario = {scenario})",
}

# Define plot names with new names
plot_names = {
    'sediment_active_percent': f"{new_dir}/Percent_Active (Scenario{scenario_num}).png",
    'num_pulse_each_link_DS': f"{new_dir}/TotalPulseParcel(Scenario{scenario_num}).png",
    'num_total_parcels_each_link_DS': f"{new_dir}/Total_parcel(Scenario{scenario_num}).png",
    'num_active_pulse_nozero_DS': f"{new_dir}/ActivePulse(Scenario{scenario_num}).png",
    'vol_pulse_nozero_DS': f"{new_dir}/Vol_pulse(Scenario{scenario_num}).png",
    'vol_nozero_DS': f"{new_dir}/Vol(Scenario{scenario_num}).png",
    'D_mean_pulse_each_link_nozero_DS': f"{new_dir}/D_mean_pulse(Scenario{scenario_num}).png",
    'D_mean_each_link_nozero_DS': f"{new_dir}/Dmean_(Scenario{scenario_num}).png",
    'percent_active_pulse_DS': f"{new_dir}/fraction_active_pulse(Scenario{scenario_num}).png",
    'percent_active_DS': f"{new_dir}/Fraction_Active (Scenario{scenario_num}).png",
    'Elev_change': f"{new_dir}/TrackElevChange(Scenario{scenario_num}).png",
    'transport_capacity': f"{new_dir}/transport_capacity(Scenario{scenario_num}).png",
}

# Define colormaps with new names
cmap = {
    'sediment_active_percent': 'viridis',
    'num_pulse_each_link_DS': 'inferno',
    'num_total_parcels_each_link_DS': 'inferno',
    'num_active_pulse_nozero_DS': 'winter_r',
    'vol_pulse_nozero_DS': 'plasma_r',
    'vol_nozero_DS': 'plasma_r',
    'D_mean_pulse_each_link_nozero_DS': 'winter_r',
    'D_mean_each_link_nozero_DS': 'winter_r',
    'percent_active_pulse_DS': 'Wistia',
    'percent_active_DS': 'Wistia',
    'Elev_change': 'coolwarm',
    'transport_capacity': 'viridis',
}

norm = {
    'sediment_active_percent': None,
    'num_pulse_each_link_DS': None,
    'num_total_parcels_each_link_DS': None,
    'num_active_pulse_nozero_DS': None,
    'vol_pulse_nozero_DS': matplotlib.colors.LogNorm(vmin=0.1, vmax=5000),
    'vol_nozero_DS': matplotlib.colors.LogNorm(vmin=0.1, vmax=5000),
    'D_mean_pulse_each_link_nozero_DS': matplotlib.colors.LogNorm(vmin=0.01, vmax=1.0),
    'D_mean_each_link_nozero_DS': matplotlib.colors.LogNorm(vmin=0.01, vmax=0.1),
    'percent_active_pulse_DS': matplotlib.colors.LogNorm(vmin=0.1, vmax=1.0),
    'percent_active_DS': matplotlib.colors.LogNorm(vmin=0.1, vmax=1.0),
    'Elev_change': colors.CenteredNorm(),
    'transport_capacity': None,
}
    
# Plot for all time
for var_name, data in variables.items():
    plt.figure()
    
    if var_name == 'Elev_change':  # Special handling for Elev_change
        plt.pcolor(time_array, dist_upstream_nodes, data, shading='auto', cmap=cmap[var_name], norm=norm[var_name])
    else:
        plt.pcolor(time_array, dist_upstream, data, shading='auto', cmap=cmap[var_name], norm=norm[var_name] if norm[var_name] is not None else None)
    
    x_right_edge = plt.gca().get_xlim()[1]
    canyonline = plt.plot(np.ones(len(canyon_reaches))*x_right_edge,dist_upstream[canyon_reaches],color='grey')
    canyonline[0].set_clip_on(False)
    plt.text(x_right_edge, 40, 'Canyon Reaches', rotation=90, va='center', ha='left', color='grey')
    pulseline = plt.plot(np.ones(len(fan_location))*x_right_edge,dist_upstream[fan_location],color='red')
    pulseline[0].set_clip_on(False)
    
    plt.text(x_right_edge, 10, "Pulse", rotation= 90, va= 'center', ha= 'left', color='red')
    plt.axis([1,timesteps,0,max(dist_upstream_nodes)])
    plt.colorbar(label=colorbar_labels[var_name])
    plt.title(plot_titles[var_name])
    plt.xlabel("Timesteps")
    plt.ylabel("Distance from upstream (km)")
    plt.savefig(plot_names[var_name], dpi= 700)
    plt.show()    

#Plot since steady state
for var_name_ss, data_ss in variables.items():
    plt.figure()
    if "pulse" not in var_name_ss:
        shortened_variable = data_ss[:, pulse_time-1:]
        steady_state_values = data_ss[:, pulse_time-1]
        normalized_variable = shortened_variable - steady_state_values[:, np.newaxis]
        if var_name_ss == 'Elev_change':  # Special handling for Elev_change
            plt.pcolor(time_array_ss, dist_upstream_nodes, normalized_variable, shading='auto', cmap=cmap[var_name_ss], norm=norm[var_name_ss])
        else:
            plt.pcolor(time_array_ss, dist_upstream, normalized_variable, shading='auto', cmap=cmap[var_name_ss], norm=norm[var_name_ss] if norm[var_name_ss] is not None else None)
    else:
        continue
    
    x_right_edge = plt.gca().get_xlim()[1]
    canyonline = plt.plot(np.ones(len(canyon_reaches))*x_right_edge,dist_upstream[canyon_reaches],color='grey')
    canyonline[0].set_clip_on(False)
    plt.text(x_right_edge, 40, 'Canyon Reaches', rotation=90, va='center', ha='left', color='grey')
    pulseline = plt.plot(np.ones(len(fan_location))*x_right_edge,dist_upstream[fan_location],color='red')
    pulseline[0].set_clip_on(False)
    plt.text(x_right_edge, 10, "Pulse", rotation= 90, va= 'center', ha= 'left', color='red')
    plt.axis([1,timesteps,0,max(dist_upstream_nodes)])    
    plt.colorbar(label=colorbar_labels[var_name_ss])
    plt.title('SS_' + plot_titles[var_name_ss])
    plt.xlabel("Timesteps")
    plt.ylabel("Distance from upstream (km)")
    plt.savefig(plot_names[var_name_ss], dpi= 700)
    plt.show()  


#what is the number of parcels active per link
plt.figure()
plt.plot(num_active_parcels_each_link_DS)
plt.title("Number of parcels active (Scenario = %i)" %scenario)
plt.xlabel("Link number downstream")
plt.ylabel("Number of active parcels")
plot_name = str(new_dir) + "/" + 'ActiveParcel(Scenario' + str(scenario_num) + ').png'
plt.savefig(plot_name, dpi=700)
plt.show()


#D50 for each link
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(dist_upstream, D50_each_link,'-', color='k')
ax2.plot(dist_upstream, num_total_parcels_each_link_DS,'-', color='brown') #,'.' makes it a dot chart, '-' makes line chart
ax1.set_xlabel("Distance from upstream (km)")
ax1.set_ylabel("D50", color='k')
plt.title("Number of parcels and the grain size (Scenario = %i)" %scenario)
ax2.set_ylabel("Number of total parcels", color='brown')
plot_name = str(new_dir) + "/" + 'RSM(Scenario' + str(scenario_num) + ').png'
plt.savefig(plot_name, dpi=700)
plt.show()


#NST plots
for parcel_sizes in range(len(active_d_mean)):
    grain_size_change = initial_d- active_d_mean[parcel_sizes]
    plt.plot(grain_size_change)
plt.show()

grid.add_field("elevation_change", elev_change_at_link, at="link", units="m", copy=True, clobber=True)
grid.add_field("change_D", grain_size_change, at="link", units="m", copy=True, clobber=True)


log_width = np.log(width)
fig = plot_network_and_parcels(grid,parcels,parcel_time_index=0,link_attribute=('elevation_change'),parcel_alpha=0,network_linewidth=log_width, link_attribute_title= "Elevation Change in each reach (m)", network_cmap= "coolwarm")
plot_name = str(new_dir) + "/" + 'NST_ELEVCHANGE(Scenario' + str(scenario_num) + ').jpg'
fig.savefig(plot_name, bbox_inches='tight', dpi=700)


# fig, ax = plt.subplots(figsize=(10, 8))
# Suiattle_Dem = np.array(Suiattle_Dem)  # Ensure `Suiattle_Dem` is a NumPy array
# im_dem = ax.imshow(Suiattle_Dem, cmap='gist_gray', vmin=np.min(Suiattle_Dem), vmax=np.max(Suiattle_Dem))
# plt.colorbar(im_dem, ax=ax, label="Elevation (m)")
# plt.title("Suiattle DEM")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")

# fig.savefig(plot_name, bbox_inches='tight', dpi=700)


fig =plot_network_and_parcels(grid,parcels,parcel_time_index=0, link_attribute=('change_D'),parcel_alpha=0, link_attribute_title="Diameter [m]", network_cmap= "coolwarm")
plot_name = str(new_dir) + "/" + 'NST_grain_size_change(Scenario' + str(scenario_num) + ').jpg'
fig.savefig(plot_name, bbox_inches='tight', dpi=700)