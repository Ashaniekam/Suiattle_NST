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

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from matplotlib.ticker import LogFormatter


import tempfile
import numpy as np
import pandas as pd
import os
import random
import pathlib
import pickle
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
from landlab.io.native_landlab import load_grid, save_grid



OUT_OF_NETWORK = NetworkModelGrid.BAD_INDEX - 1

# #### Selecting abrasion/density scenario #####
scenario = 1

if scenario == 1:
    scenario_num = "none"
    model_state = "no_abrasion"
elif scenario == 2:
    scenario_num = "SHRS proxy"
    model_state = "SHRS proxy"
else: 
    scenario_num = "4_times SHRS proxy"
    model_state = "4_times SHRS proxy"

# ##### Basic model parameters 

timesteps = 70
pulse_time = 2
dt = 60*60*24*1  # len of timesteps (1 day)

n_lines = 10 # note: # timesteps must be divisible by n_lines with timesteps%n_lines == 0
color = iter(plt.cm.viridis(np.linspace(0, 1, n_lines+1)))
c = next(color)


# bookkeeping 
new_dir_name = "Pulse location test" + model_state ## Use when testing
#new_dir_name = "Longer_Run_Test_Scenario_1" + model_state

new_dir = pathlib.Path(os.getcwd(), new_dir_name)
new_dir.mkdir(parents=True, exist_ok=True)
output_folder = os.path.join(os.getcwd (), ("Results-"+ new_dir_name))
if not os.path.exists(output_folder):
    os.mkdir(output_folder)


# %% ##### Set up the grid, channel characteristics #####
shp_file = os.path.join(os.getcwd (), ("Suiattle_river.shp"))
points_shapefile = os.path.join(os.getcwd(), ("Suiattle_nodes.shp"))


grid = read_shapefile(
    shp_file,
    points_shapefile = points_shapefile,
    node_fields = ["usarea_km2", "F1m_LiDAR"],
    link_fields = ["usarea_km2", "Length_m", "Slope_sm", "Width_sm"],
    link_field_conversion = {"usarea_km2": "drainage_area", "Slope_sm":"channel_slope", "Width_sm": "channel_width", "Length_m":"reach_length", }, #, "Width": "channel_width";  how to read in channel width
    node_field_conversion = {"usarea_km2": "drainage_area", "F1m_LiDAR": "topographic__elevation"},
    threshold = 0.1,
    )


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

discharge = (0.2477 * area_link) + 8.8786
discharge_DS= discharge[index_sorted_area_link]
#Figure and justification in "Discharge from DHSVM Suiattle_exploring.xlsx"

#Sorted Topographic Elevation and Bedrock Elevation
topo = grid.at_node['topographic__elevation'].copy()
topo[topo== 776.32598877] += 3 #attemp to smooth the grid to get rid of bottleneck
topo_DS = topo[index_sorted_area_node]
grid.at_node["bedrock__elevation"] = topo.copy()

#Sorted Width
width = grid.at_link["channel_width"]
width_DS = width[index_sorted_area_link]

#Sorted Length
length = grid.at_link["reach_length"]
length_DS = length[index_sorted_area_link]

#Sorted Slope
slope = grid.at_link["channel_slope"]
slope_DS = slope[index_sorted_area_link]
initial_slope= slope.copy()



grid.at_link["dist_downstream"]= np.cumsum(length_DS)
dist_downstream = grid.at_link["dist_downstream"]/1000
dist_downstream_nodes = np.insert(dist_downstream, 0, 0.0)

#to calculate flow depth
Mannings_n = 0.086 #median calculated from Suiattle gages (units m^3/s)
grid.at_link["flow_depth"] = ((Mannings_n*discharge)/ ((slope**0.5)*width))**0.6

depth = grid.at_link["flow_depth"].copy()
depth_DS = depth[index_sorted_area_link]

# %% Set up parcels 
rho_water = 1000
rho_sed = 2650
gravity = 9.81
tau = rho_water * gravity * depth * slope
tau_DS = tau[index_sorted_area_link]

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
plt.plot(dist_downstream, slope_DS, '-' )
plt.title("Starting channel Slope")
plt.ylabel("Slope")
plt.xlabel("Distance downstream (km)")
plt.show()

#FLOW DEPTH
plt.figure()
plt.plot(dist_downstream, depth_DS, '-' )
plt.title("Flow Depth")
plt.ylabel("Flow depth (m)")
plt.xlabel("Distance downstream (km)")
plt.show()

# #Channel Width
plt.figure()
plt.plot(dist_downstream, width_DS, '-' )
plt.title("Channel Width")
plt.ylabel("Channel Width (m)")
plt.xlabel("Distance downstream (km)")
plt.show()

tau_star = tau/((rho_sed - rho_water) * gravity * D50_each_link)

# Plot for shield stress
plt.figure()
plt.title("Tau_star")
plt.plot(dist_downstream, tau_star)
plt.ylabel("Shield Stress")
plt.xlabel("Distance downstream (km)")
plt.show()

# D50 for each link
#find line of best fit
a, b = np.polyfit(dist_downstream, D50_each_link, 1)
str_tau_c_multiplier = str(tau_c_multiplier)
plt.figure()
plt.scatter(dist_downstream, D50_each_link)
plt.plot(dist_downstream, a*dist_downstream+b)  
plt.ylabel("D50 each link")
plt.title("d50 each link (tau_c_multiplier = " + str_tau_c_multiplier +")")
plt.xlabel("Distance downstream (km)")
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
    k_transp_dep_abr= 15.0,
)

# %% Setup for plotting during/after loop

#variables needed for plots
Pulse_cum_transport = []

# Tracked for each timestep
n_recycled_parcels = np.ones(timesteps)*np.nan
vol_pulse_left = np.ones(timesteps)*np.nan
vol_pulse_on = np.ones(timesteps)*np.nan
percent_pulse_added = np.ones(timesteps)*np.nan
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
Sed_flux= np.empty([grid.number_of_links, timesteps])
Sand_fraction_active = np.ones([grid.number_of_links,timesteps])*np.nan

Elev_change = np.empty([grid.number_of_nodes,timesteps]) # 2d array of elevation change in each timestep 


# %% Pulse element variables
fan_thickness = np.array([3.0, 6.25, 6.25, 5.75, 2.75, 1.25]) #meters
fan_location= np.array([6, 7, 8, 9, 10, 11])
pulse_parcel_vol = 10 #volume of each parcel m^3
total_num_pulse_parcels = 490000


initial_pulse_volume = fan_thickness*length[fan_location]*width[fan_location] # m^3; volume of pulse that should be added to each link of the Chocolate Fan
initial_pulse_rock_volume = initial_pulse_volume * (1-bed_porosity)
initial_num_pulse_parcels_by_vol = initial_pulse_rock_volume/pulse_parcel_vol #number of parcels added to each link based on volume; 
total_num_initial_pulse_parcels = int(np.sum(initial_num_pulse_parcels_by_vol))


num_pulse_parcels_in_network = total_num_initial_pulse_parcels


############# distribute the total_num_pulse_parcels proportional to fan thickness instead of evenly -- needs clean up
fan_proportions = fan_thickness / fan_thickness.sum()

# Distribute total_num_pulse_parcels according to the proportions
pulse_each_fan_link = (fan_proportions * total_num_pulse_parcels).astype(int) # I am making 490000 parcels

# Calculate the remainder due to integer conversion
remainder = total_num_pulse_parcels - pulse_each_fan_link.sum()

# Add the remainder to the largest element in pulse_each_link
pulse_each_fan_link[np.argmax(fan_proportions)] += remainder


pulse_location = [index for index, freq in enumerate((pulse_each_fan_link).astype(int), start=6) for _ in range(freq)]

random.shuffle(pulse_location)

newpar_element_id = pulse_location
newpar_element_id = np.expand_dims(newpar_element_id, axis=1)   


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
newpar_grid_elements = np.array(
    np.empty(
        (np.shape(newpar_element_id)), dtype=object
    )
)

newpar_grid_elements.fill("link")

# %% grain size for Pulse
#dummy_GSD= np.random.lognormal(np.log(0.05),np.log(5.6), len(newpar_element_id)*2) 

field_grain_sizes = pd.read_excel((os.path.join(os.getcwd (), ("SuiattleFieldData_Combined20182019.xlsx"))), sheet_name = '3 - Grain size df deposit', header = 0)
Suiattle_gsd = field_grain_sizes["Size (cm)"].values
Suiattle_gsd_m = Suiattle_gsd/100

scaled_size = total_num_pulse_parcels * 10

# Scale up by repeating the array
scaled_gsd = np.resize(Suiattle_gsd_m, scaled_size)
sorted_scaled_gsd = np.sort(scaled_gsd)
scaled_count = np.linspace(0,100,len(scaled_gsd))

######



num_rows = np.shape(newpar_element_id)[0]

new_D = np.empty(((np.shape(newpar_element_id)[0]), 1), dtype=float)

# Filter values in GSD that are less than 0.5
truncated_GSD = scaled_gsd[scaled_gsd < 0.5]
#truncated_GSD = grain_sizes[grain_sizes < 0.5]

new_D[:,0] = truncated_GSD[:total_num_pulse_parcels]
total_num_pulse_moved = 0


count = np.linspace(0,100,len(Suiattle_gsd_m))



plt.semilogx(np.sort(Suiattle_gsd_m),count, color ="blue")
plt.plot(np.log(sorted_scaled_gsd),scaled_count, color= "red")
plt.xlabel('Grain Size (log scale)')
plt.ylabel('Cumulative % Finer')
plt.show()

# unique_field_grain_sizes, count_field_sizes = np.unique(Suiattle_gsd_m, return_counts=True)
# cumulative_field_distributions = np.cumsum(count_field_sizes)

# unique_model_grain_sizes, count_model_sizes = np.unique(dummy_GSD, return_counts=True)
# cumulative_model_distributions = np.cumsum(count_model_sizes)


# plt.figure(figsize=(8, 6))
# plt.plot(unique_field_grain_sizes, cumulative_field_distributions, linestyle='-', color='red' )


# #plt.plot(unique_model_grain_sizes, cumulative_model_distributions, linestyle='-', color='blue' )

# # Set the x-axis to logarithmic scale
# plt.xscale('log')

# # Add labels and title
# plt.text(1,5, "0.06 and 4.8")
# plt.xlabel('Grain Size (log scale)')
# plt.ylabel('Cumulative Grain Size Distribution')
# plt.title('Cumulative Grain Size Distribution')
#plt.grid(which='both', linestyle='--', linewidth=0.5)

# Show the plot
#plt.show()

# %% Abrasion scenarios using Allison's field data
field_data = pd.read_excel((os.path.join(os.getcwd (), ("SuiattleFieldData_Combined20182019.xlsx"))), sheet_name = '4-SHRSdfDeposit', header = 0)
SHRS = field_data['SHRS median'].values
SHRS_MEAN = np.mean(SHRS)
SHRS_STDEV = np.std(SHRS)

SHRS_distribution = np.random.normal(SHRS_MEAN, SHRS_STDEV, (np.size(newpar_element_id)))

SHRS_distribution[SHRS_distribution<20]=20 # 9/10/24 - Hard coding this to prevent occasional VERY weak values. Min set to just below observed min (SHRS = 24)

measured_alphas = np.exp(1.122150830896329)*np.exp(-0.13637476779600016 *SHRS_distribution) # units: 1/km
 

tumbler_2 = 2 * measured_alphas
tumbler_4 = 4 * measured_alphas


if scenario == 1:
    new_abrasion_rate = 0 * np.ones(np.size(newpar_element_id))
    new_density = 2650 * np.ones(np.size(newpar_element_id))  # (kg/m3) standard
    print('Scenario 1 pulse: no abrasion')
elif scenario == 2:
    new_abrasion_rate = (measured_alphas/1000)* np.ones(np.size(newpar_element_id)) #0.3 % mass loss per METER
    new_density = 928.6259337721524*np.log(SHRS_distribution) + (-1288.1880919573282)
    #new_density = 894.978992640976*np.log(SHRS_distribution)-1156.7599235065895
    print('Scenario 2 pulse: variable abrasion, SH proxy')
else: 
    new_abrasion_rate = (tumbler_4/1000)* np.ones(np.size(newpar_element_id)) #tumbler correction 4 !!!!
    new_density = 928.6259337721524*np.log(SHRS_distribution) + (-1288.1880919573282)
    print('Scenario 3 pulse: variable abrasion, 4 * SH proxy')
    #new_density = 894.978992640976*np.log(SHRS_distribution)-1156.7599235065895

flat_new_D = new_D.reshape(-1)
new_abrasion_rate[flat_new_D < 0.002] = 0

# %% Animation start
# Initiate an animation writer using the matplotlib module, `animation`.
figAnim, axAnim = plt.subplots(1, 1)
writer = animation.FFMpegWriter(fps=6)

gif_name = output_folder +"/"+ new_dir_name + "--Elev_through_time.gif"
writer.setup(figAnim,gif_name)

# %% Model runs

for t in range(0, (timesteps*dt), dt):
    #by index of original sediment
    start_time = model_time.time()
    
    day_timestep = t/(60*60*24)
    if day_timestep % 20 == 0:
        discharge = discharge * 2.5
        print("Used the 0.1% exceedence flow for discharge")
    elif day_timestep == 50:
        discharge = discharge + 2.8316847 # 100 cfs (in m3/sec) the approx. change in discharge during an Outburst flood (USGS stream gage data for Sep 3, 2017)
        print("Discharge prepresents outburst flood")
    else:
        discharge = (0.2477 * area_link) + 8.8786
    
    
    # Masking parcels
    #boolean mask of parcels that have left the network
    mask1 = parcels.dataset.element_id.values[:,-1] == OUT_OF_NETWORK
    #boolean mask of parcels that are initial bed parcels (no pulse parcels)
    mask2 = parcels.dataset.source == "initial_bed_sed" 
    
    bed_sed_out_network = parcels.dataset.element_id.values[mask1 & mask2, -1] == -2
    
    
    
# %%    # ###########   Parcel recycling by drainage area    ###########
    
    #index of the bed parcels that have left the network
    index_exiting_parcels = np.where(bed_sed_out_network == True) 
    
    #number of parcels that will be recycled
    num_exiting_parcels = np.size(index_exiting_parcels)

    # Calculate the total sum of area_link
    total_area = sum(area_link)

    # Calculate the proportions based on drainage area
    recycled_proportions = area_link/total_area

    # Calculate the number of parcels for each link based on proportions
    num_recycle_bed = [int(recycled_proportion * num_exiting_parcels) for recycled_proportion in recycled_proportions]

    # Calculate the remaining parcels not accounted for by using int above
    remaining_bed_sed = num_exiting_parcels - sum(num_recycle_bed)

    # Calculate the number of additional parcels to add to each proportion
    additional_bed_sed = np.round(np.array(recycled_proportions) * remaining_bed_sed).astype(int)

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
    
    
    
# %% ########## Making a pulse #############  
    mask_ispulse = parcels.dataset.source == 'pulse'

    # Calculate the current timestep
    current_timestep = t // dt
    
    if total_num_pulse_parcels > num_pulse_parcels_in_network :

    # Check if it's the time for pulse evaluation
        if current_timestep >= pulse_time and (current_timestep - pulse_time) % 5 == 0:
    
            if t == dt * pulse_time:  
                print ("# First pulse time")
            
                mask_pulse = np.full(len(newpar_element_id),False)
                mask_pulse = mask_pulse[:total_num_initial_pulse_parcels]==True
                
                
                
                
                
                new_parcels = {"grid_element": newpar_grid_elements[:total_num_initial_pulse_parcels],
                      "element_id": newpar_element_id[:total_num_initial_pulse_parcels]}

                new_variables = {
                    "starting_link": (["item_id"], new_starting_link[:total_num_initial_pulse_parcels]),
                    "abrasion_rate": (["item_id"], new_abrasion_rate[:total_num_initial_pulse_parcels]),
                    "density": (["item_id"], new_density[:total_num_initial_pulse_parcels]),
                    "source": (["item_id"], new_source[:total_num_initial_pulse_parcels]),
                    "time_arrival_in_link": (["item_id", "time"], new_time_arrival_in_link[:total_num_initial_pulse_parcels]),
                    "active_layer": (["item_id", "time"], new_active_layer[:total_num_initial_pulse_parcels]),
                    "location_in_link": (["item_id", "time"], new_location_in_link[:total_num_initial_pulse_parcels]),
                    "D": (["item_id", "time"], new_D[:total_num_initial_pulse_parcels]),
                    "volume": (["item_id", "time"], new_volume[:total_num_initial_pulse_parcels]),
                
        
                    }
                
                
            
                
                
                total_num_added_pulse_parcels = total_num_initial_pulse_parcels
                

            else:
                print(" OTHER Pulses")
                num_pulse_in_fan = 0 # this timestep
                
                # Loop through pulse parcels to check their location
                for i in nst.current_link[mask_ispulse]:
                    if i in fan_location:
                        num_pulse_in_fan += 1
                
                num_new_pulse = int(total_num_initial_pulse_parcels-num_pulse_in_fan)
                
                print('The number of new pulse parcels added =', num_new_pulse)
                # Update mask_pulse to include the correct number of parcels
                mask_pulse = np.full(len(newpar_element_id),False)
                mask_pulse = mask_pulse[total_num_added_pulse_parcels:total_num_added_pulse_parcels+num_new_pulse]
                
                
                
                
                new_parcels = {"grid_element": newpar_grid_elements[total_num_added_pulse_parcels:total_num_added_pulse_parcels+num_new_pulse],
                      "element_id": newpar_element_id[total_num_added_pulse_parcels:total_num_added_pulse_parcels+num_new_pulse]}

                new_variables = {
                    "starting_link": (["item_id"], new_starting_link[total_num_added_pulse_parcels:total_num_added_pulse_parcels+num_new_pulse]),
                    "abrasion_rate": (["item_id"], new_abrasion_rate[total_num_added_pulse_parcels:total_num_added_pulse_parcels+num_new_pulse]),
                    "density": (["item_id"], new_density[total_num_added_pulse_parcels:total_num_added_pulse_parcels+num_new_pulse]),
                    "source": (["item_id"], new_source[total_num_added_pulse_parcels:total_num_added_pulse_parcels+num_new_pulse]),
                    "time_arrival_in_link": (["item_id", "time"], new_time_arrival_in_link[total_num_added_pulse_parcels:total_num_added_pulse_parcels+num_new_pulse]),
                    "active_layer": (["item_id", "time"], new_active_layer[total_num_added_pulse_parcels:total_num_added_pulse_parcels+num_new_pulse]),
                    "location_in_link": (["item_id", "time"], new_location_in_link[total_num_added_pulse_parcels:total_num_added_pulse_parcels+num_new_pulse]),
                    "D": (["item_id", "time"], new_D[total_num_added_pulse_parcels:total_num_added_pulse_parcels+num_new_pulse]),
                    "volume": (["item_id", "time"], new_volume[total_num_added_pulse_parcels:total_num_added_pulse_parcels+num_new_pulse]),
                
        
                    }
                
                
                
                
                
                total_num_added_pulse_parcels += num_new_pulse
               

    
        
    
            new_source = np.array(new_source)
            percent_pulse_added[np.int64(t/dt)] = total_num_added_pulse_parcels/total_num_pulse_parcels
            print('making a pulse')
            print("The number of pulse parcels added should be: ", np.shape(mask_pulse))
            print("On this timestep the total number of pulse parcels added = ", total_num_added_pulse_parcels)
            
            print('total_num_added_pulse_parcels =', total_num_added_pulse_parcels)
            print('t = ',t, ',timestep = ', t/dt) 
        
        # %% Add pulse         
            # new_parcels = {"grid_element": newpar_grid_elements[mask_pulse],
            #       "element_id": newpar_element_id[mask_pulse]}

            # new_variables = {
            #     "starting_link": (["item_id"], new_starting_link[mask_pulse]),
            #     "abrasion_rate": (["item_id"], new_abrasion_rate[mask_pulse]),
            #     "density": (["item_id"], new_density[mask_pulse]),
            #     "source": (["item_id"], new_source[mask_pulse]),
            #     "time_arrival_in_link": (["item_id", "time"], new_time_arrival_in_link[mask_pulse]),
            #     "active_layer": (["item_id", "time"], new_active_layer[mask_pulse]),
            #     "location_in_link": (["item_id", "time"], new_location_in_link[mask_pulse]),
            #     "D": (["item_id", "time"], new_D[mask_pulse]),
            #     "volume": (["item_id", "time"], new_volume[mask_pulse]),
            
    
            #     }
 
            parcels.add_item(
                    time=[nst._time],
                    new_item = new_parcels,
                    new_item_spec = new_variables
                    )
        
            #percent_pulse_remain = (vol_pulse_on/(np.sum(initial_pulse_rock_volume)))*100
    else:
        continue
# %% Run one timestep    
    
    # #######   Ta da!    #########     
    nst.run_one_step(dt)

    
#%% #Calculate Variables and tracking timesteps
    if t == parcels.dataset.time[0].values: # if we're at the first timestep

        avg_init_sed_thickness = np.mean(
            grid.at_node['topographic__elevation'][:-1]-grid.at_node['bedrock__elevation'][:-1])        
        grid.at_node['bedrock__elevation'][-1]=grid.at_node['bedrock__elevation'][-1]+avg_init_sed_thickness


        elev_initial = grid.at_node['topographic__elevation'].copy()
        
        
    Elev_change[:,np.int64(t/dt)] = grid.at_node['topographic__elevation']-elev_initial
    

    if t%((timesteps*dt)/n_lines)==0:  
        
        # Animation
        plt.figure(figAnim)
        
    
        elev_change = grid.at_node['topographic__elevation']-elev_initial
        
        plt.plot(dist_downstream_nodes, elev_change, c= "darkblue")
        plt.xlabel('Distance downstream (km)')
        plt.title("Elevation change through time")
        plt.ylabel('Elevation change (m)')
        plt.ylim(-7, 10)
        text = plt.text(35,7,str(np.round(t/(60*60*24),1))+" Days")
      
        writer.grab_frame()
    
        # shade it out for next frame, clear text
        text.remove()
        plt.plot(dist_downstream_nodes, Elev_change, c = 'w', alpha = 0.2)
        plt.pause(0.001)
    
    

    #Tracking timesteps
    print("Model time: ", t/(60*60*24), " 4 hours passed (", t/dt, 'timesteps)')
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
    active_parcel_element_id = parcels.dataset.element_id.values[mask_active,-1].astype(int) #array of active parcels' element IDs
    parcel_volume = parcels.dataset.volume.values[:, -1] #parcel volume
    active_layer_volume = nst._active_layer_thickness *width*length
    
    weighted_sum_velocity = np.full(number_of_links, np.nan)
    sorted_indices = np.argsort(active_parcel_element_id)
    
    weighted_sum_velocity = np.zeros(grid.number_of_links)*np.nan
    for link in range(grid.number_of_links):
        mask_here = parcels.dataset.element_id.values[:,-1] == link
        
        parcel_velocity = nst._pvelocity[mask_here & mask_active]
        parcel_vol = parcels.dataset.volume.values[mask_here & mask_active,-1]
        weighted_sum_velocity[link] = np.sum(parcel_velocity * parcel_vol)
    weighted_mean_velocity = weighted_sum_velocity / active_layer_volume #divide that by active layer volume
    # Convert weighted-mean velocity to m/s
    Sed_flux [:,np.int64(t/dt)]= weighted_mean_velocity * nst._active_layer_thickness * grid.at_link['channel_width']
    
    Sand_fraction_active[:,np.int64(t/dt)] = grid.at_link["sediment__active__sand_fraction"]
    
    # Calculating volume of pulse that remains after abrasion    
    volume_of_pulse = parcels.dataset.volume.values[mask_ispulse,-1]

    vol_pulse_on[np.int64(t/dt)]=np.sum(volume_of_pulse[element_pulse!= OUT_OF_NETWORK])
    vol_pulse_left[np.int64(t/dt)]=np.sum(volume_of_pulse[element_pulse == OUT_OF_NETWORK])
     
    #percent_pulse_remain = (vol_pulse_on/(vol_pulse_on + vol_pulse_left))*100
     
    
#%%  #Aggregators for all parcels
    
    # Populating arrays with data from this timestep. 
    sed_act_vol= grid.at_link["sediment__active__volume"][index_sorted_area_link] 
  
    sed_total_vol[:,np.int64(t/dt)] = grid.at_link["sediment_total_volume"][index_sorted_area_link]
    
    #### TRYING TO GET RID OF WARNINGS
    # Create a mask where sed_total_vol is non-zero
    mask_sed_vol = sed_total_vol[:, np.int64(t/dt)] != 0

    # Initialize sediment_active_percent with zeros (to handle the cases where both arrays are zero)
    sediment_active_percent[:, np.int64(t/dt)] = 0

    # Perform division only where the mask is True (i.e., sed_total_vol is not zero)
    sediment_active_percent[mask_sed_vol, np.int64(t/dt)] = sed_act_vol[mask_sed_vol] / sed_total_vol[mask_sed_vol, np.int64(t/dt)]


    #sediment_active_percent[:,np.int64(t/dt)]= sed_act_vol/sed_total_vol[:,np.int64(t/dt)]
    
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
        # Pulse_cum_transport = np.append(
        #          Pulse_cum_transport,
        #          np.expand_dims(
        #                  nst._distance_traveled_cumulative[-total_num_added_pulse_parcels:],
        #                  axis = 1),
        #          axis = 1
        #          )
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

    #thickness_at_link = volume_pulse_at_each_link[:,2]/grid.at_link["channel_width"]/grid.at_link["reach_length"]/(1-bed_porosity)

# %% Animation

plt.figure(figAnim)
plt.close()

writer.finish()

# %% Pickle

# #Pickle the whole NST!
# nst_path = output_folder+"/"+new_dir_name+'--nst.pickle'

# with open(nst_path, 'wb') as file:
#     # Serialize and write the variable to the file
#     pickle.dump(nst, file)

# parcel_path = output_folder+"/"+new_dir_name+'--parcels.pickle'

# with open(parcel_path, 'wb') as file:
#     # Serialize and write the variable to the file
#     pickle.dump(parcels, file)

#Editing and saving variables in DataRecord and csv file
# parcel_data = parcels.dataset.to_dataframe()
# parcel_data.to_csv('Parcels_saved_data.csv', sep=',', index=True)

## Note to self: open these via...

# with open(file_path, 'rb') as file:
#     parcels = pickle.load(file)


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

pulse_ids = parcels.dataset.element_id.values[mask_ispulse]

# Replace NaNs with zeros and create a mask where the denominator is non-zero
mask_active_pulse = np.nan_to_num(num_pulse_each_link, nan=0.0) != 0

# Perform the division where valid, otherwise set to zero
percent_active_pulse = np.divide(num_active_pulse, num_pulse_each_link, where=mask_active_pulse, out=np.zeros_like(num_active_pulse, dtype=float))

percent_active_pulse[np.isnan(num_pulse_each_link)] = np.nan

percent_active_pulse_DS = percent_active_pulse[index_sorted_area_link] 
percent_active_pulse_DS[percent_active_pulse_DS == 0]= np.nan



# Create a mask where the denominator is non-zero
mask_active_parcels = num_each_link != 0

# Initialize percent_active with zeros (for cases where division is invalid)
percent_active = np.zeros_like(num_active_parcels_each_link, dtype=float)

# Perform division only where num_each_link is non-zero
percent_active[mask_active_parcels] = num_active_parcels_each_link[mask_active_parcels] / num_each_link[mask_active_parcels]

percent_active_DS = percent_active[index_sorted_area_link] 
percent_active_DS[percent_active_DS == 0]= np.nan
#%% Editing and saving variables in DataRecord and text file
parcels_dataframe= parcels.dataset.to_dataframe()
parcels_dataframe.to_csv('Parcels_data_scenario.csv', sep=',', index=True) 

#Saving a text file of model characteristics
timestep_in_days = dt/86400
# Converting the time in seconds to a timestamp
#c_ti = model_time.ctime(os.path.getctime("NST_Suiattle_Before pickel_Nov14.py"))
text_file = os.path.join(new_dir, 'TESTING ('+scenario_num+ ').txt')   
file = open('Suiattle_run_characteristics'+scenario_num+ ').txt', 'w')
model_characteristics = ["This code is running for scenario %s" %scenario,  "This is the tau_c_multiplier %s" %tau_c_multiplier, 
                          "This is the number of timesteps %s" %timesteps, "The number of days in each timestep is %s" % timestep_in_days, # length of dt in days
                          "The pulse is added at timestep %s" %pulse_time,
                          "The volume of each pulse parcel is %s" %pulse_parcel_vol, "This is the number of pulse parcels added to the network %s"
                          %num_pulse_parcels_in_network,
                          
                          "The current stage of this code is editing variables to be normalized since steady state."]
for line in model_characteristics:
    file.write(line)
    file.write('\n')
file.close()

# # %% ####   Plotting and analysis of results    ############  

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

# Grain size distribution of the pulse material #MIGHT WANT THIS FIGURE...
# plt.figure()
# bins = np.logspace(-4,2,50)
# plt.xscale('log')
# plt.hist(Pulse starting gsd?,bins=bins)

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
travel_dist = nst._distance_traveled_cumulative[:-num_pulse_parcels_in_network]
nonzero_travel_dist = travel_dist[travel_dist>0]
plt.hist(np.log10(nonzero_travel_dist),30) # better to make log-spaced bins...
plt.xlabel('log10( Cum. parcel travel distance (m) )')
plt.ylabel('Number of non-pulse parcels')
plt.title("How far have the non-pulse parcels traveled in total? (Scenario = %i)" %scenario)
plt.savefig(plot_name, dpi=700)
plt.text(2,50, 'n = '+str(len(nonzero_travel_dist))+' parcels')
plt.text(2,50, 'n = '+str(len(nonzero_travel_dist))+' parcels')

plt.figure(dpi=600)
plt.pcolor(time_array, dist_downstream, Sand_fraction_active, 
            shading='nearest', 
            norm= None, 
            cmap="copper"
            ) 
plt.colorbar(label= "Sand fraction")
plt.xlabel('Days')
plt.ylabel('Distance downstream (km)')
plot_name = str(new_dir) + "/" + 'Sand_frac(Scenario' + str(scenario_num) + ').png'
plt.savefig(plot_name)
plt.show()

plt.figure(dpi=600)
plt.plot(timestep_array, percent_pulse_added, '.') 
plt.xlabel('Days')
plt.ylabel('Fraction of Pulse added')
plot_name = str(new_dir) + "/" + 'percentage of the pulse added (Scenario' + str(scenario_num) + ').png'
plt.savefig(plot_name)
plt.show()



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
    'transport_capacity': Sed_flux,
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
        mesh = plt.pcolor(time_array, dist_downstream_nodes, data, shading='auto', cmap=cmap[var_name], norm=norm[var_name])
        
    else:
        mesh = plt.pcolor(time_array, dist_downstream, data, shading='auto', cmap=cmap[var_name], norm=norm[var_name] if norm[var_name] is not None else None)
        
    
    x_right_edge = plt.gca().get_xlim()[1]
    canyonline = plt.plot(np.ones(len(canyon_reaches))*x_right_edge,dist_downstream[canyon_reaches],color='grey')
    canyonline[0].set_clip_on(False)
    plt.text(x_right_edge, 40, 'Canyon Reaches', rotation=90, va='center', ha='left', color='grey')
    pulseline = plt.plot(np.ones(len(fan_location))*x_right_edge,dist_downstream[fan_location],color='red')
    pulseline[0].set_clip_on(False)
    
    plt.text(x_right_edge, 10, "Pulse", rotation= 90, va= 'center', ha= 'left', color='red')
    plt.axis([1,timesteps,0,max(dist_downstream_nodes)])
    plt.axvline(x=pulse_time, color='gray', linestyle='--', linewidth=1.5, label=f'Day {pulse_time}')

    
    mesh.set_array(data)
    plt.colorbar(mesh, label=colorbar_labels[var_name])
    plt.title(plot_titles[var_name])
    plt.xlabel("Days")
    plt.ylabel("Distance downstream (km)")
    plt.savefig(plot_names[var_name], dpi= 700)
    plt.show()    

# #Plot since steady state
# for var_name_ss, data_ss in variables.items():
#     plt.figure()
#     if "pulse" not in var_name_ss:
#         shortened_variable = data_ss[:, pulse_time-1:]
#         steady_state_values = data_ss[:, pulse_time-1]
#         normalized_variable = shortened_variable - steady_state_values[:, np.newaxis]
#         if var_name_ss == 'Elev_change':  # Special handling for Elev_change
#             mesh_ss = plt.pcolor(time_array_ss, dist_downstream_nodes, normalized_variable, shading='auto', cmap=cmap[var_name_ss], norm=norm[var_name_ss])
#         else:
#             mesh_ss = plt.pcolor(time_array_ss, dist_downstream, normalized_variable, shading='auto', cmap=cmap[var_name_ss], norm=norm[var_name_ss] if norm[var_name_ss] is not None else None)
#     else:
#         continue
    
#     x_right_edge = plt.gca().get_xlim()[1]
#     canyonline = plt.plot(np.ones(len(canyon_reaches))*x_right_edge,dist_downstream[canyon_reaches],color='grey')
#     canyonline[0].set_clip_on(False)
#     plt.text(x_right_edge, 40, 'Canyon Reaches', rotation=90, va='center', ha='left', color='grey')
#     pulseline = plt.plot(np.ones(len(fan_location))*x_right_edge,dist_downstream[fan_location],color='red')
#     pulseline[0].set_clip_on(False)
#     plt.text(x_right_edge, 10, "Pulse", rotation= 90, va= 'center', ha= 'left', color='red')
#     plt.axis([1,timesteps,0,max(dist_downstream_nodes)])  
#     mesh_ss.set_array(normalized_variable)
#     plt.colorbar(mesh_ss, label=colorbar_labels[var_name_ss])
#     plt.title('SS_' + plot_titles[var_name_ss])
#     plt.xlabel("Timesteps")
#     plt.ylabel("Distance downstream  (km)")
#     plt.show()  


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
ax1.plot(dist_downstream, D50_each_link,'-', color='k')
ax2.plot(dist_downstream, num_total_parcels_each_link_DS,'-', color='brown') #,'.' makes it a dot chart, '-' makes line chart
ax1.set_xlabel("Distance downstream (km)")
ax1.set_ylabel("D50", color='k')
plt.title("Number of parcels and the grain size (Scenario = %i)" %scenario)
ax2.set_ylabel("Number of total parcels", color='brown')
plot_name = str(new_dir) + "/" + 'RSM(Scenario' + str(scenario_num) + ').png'
plt.savefig(plot_name, dpi=700)
plt.show()

plt.figure()
plt.plot(area_link_DS, discharge_DS, '.' )
plt.ylabel("Discharge ($m^3$/sec)")
plt.xlabel("Drainage Area ($km^2$)")
plt.show()

# # #NST plots
# # for parcel_sizes in range(len(active_d_mean)):
# #     grain_size_change = initial_d- active_d_mean[parcel_sizes]
# #     plt.plot(grain_size_change)
# # plt.show()

# # grid.add_field("elevation_change", elev_change_at_link, at="link", units="m", copy=True, clobber=True)
# # grid.add_field("change_D", grain_size_change, at="link", units="m", copy=True, clobber=True)


# # log_width = np.log(width)
# # fig = plot_network_and_parcels(grid,parcels,parcel_time_index=0,link_attribute=('elevation_change'),parcel_alpha=0,network_linewidth=log_width, link_attribute_title= "Elevation Change in each reach (m)", network_cmap= "coolwarm")
# # plot_name = str(new_dir) + "/" + 'NST_ELEVCHANGE(Scenario' + str(scenario_num) + ').jpg'
# # fig.savefig(plot_name, bbox_inches='tight', dpi=700)


# # #NST plots

# # grid.add_field("elevation_change", elev_change_at_link, at="link", units="m", copy=True, clobber=True)
# # log_width = np.log(width)

# # fig, ax = plt.subplots(figsize=(10, 8))
# # Suiattle_Dem = np.array(Suiattle_Dem)  # Ensure `Suiattle_Dem` is a NumPy array
# # im_dem = ax.imshow(Suiattle_Dem, cmap='gist_gray', vmin=0, vmax=np.max(Suiattle_Dem))
# # ax= plot_network_and_parcels(grid,parcels,parcel_time_index=0,link_attribute=('elevation_change'),parcel_alpha=0,network_linewidth=log_width, link_attribute_title= "Elevation Change in each reach (m)", network_cmap= "coolwarm")
# # plt.colorbar(im_dem, ax=ax, label="Elevation (m)")
# # plt.title("Suiattle DEM")
# # plt.xlabel("Longitude")
# # plt.ylabel("Latitude")


# # fig, ax = plt.subplots(figsize=(10, 8))
# # Suiattle_Dem = np.array(Suiattle_Dem)  # Ensure `Suiattle_Dem` is a NumPy array
# # im_dem = ax.imshow(Suiattle_Dem, cmap='gist_gray', vmin=0, vmax=np.max(Suiattle_Dem))
# # plt.colorbar(im_dem, ax=ax, label="Elevation (m)")
# # plt.title("Suiattle DEM")
# # plt.xlabel("Longitude")
# # plt.ylabel("Latitude")

# # fig.savefig(plot_name, bbox_inches='tight', dpi=700)

# plt.figure()
# plt.hist(np.log10(SHRS_proxy), color = "blue", label= "SHRS Proxy")
# plt.hist(np.log10(Double_SHRS), color= "red", alpha= 0.7, label= "Double SHRS Proxy")
# plt.ylabel("Number of pulse parcels")
# plt.xlabel("Log10 Abrasion rate (1/m)")
# plt.legend()


# # fig =plot_network_and_parcels(grid,parcels,parcel_time_index=0, link_attribute=('change_D'),parcel_alpha=0, link_attribute_title="Diameter [m]", network_cmap= "coolwarm")
# # plot_name = str(new_dir) + "/" + 'NST_grain_size_change(Scenario' + str(scenario_num) + ').jpg' # fig.savefig(plot_name, bbox_inches='tight', dpi=700)