# -*- coding: utf-8 -*-
"""
Spyder Editor

This is the code for my thesis on the Suiattle River


   
@author: longrea, pfeiffea

"""
# %% import
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation

import numpy as np
import pandas as pd
import os
import random
import pathlib
import xarray as xr
import time as model_time


from landlab.components import (
    FlowDirectorSteepest,
    NetworkSedimentTransporter,
    BedParcelInitializerDepth,
)
from landlab.io import read_shapefile
from landlab.grid.network import NetworkModelGrid

from landlab.data_record.aggregators import aggregate_items_as_mean
from landlab.data_record.aggregators import aggregate_items_as_sum
from landlab.data_record.aggregators import aggregate_items_as_count

OUT_OF_NETWORK = NetworkModelGrid.BAD_INDEX - 1

# %% Basic model setup and knobs 

# #### Selecting abrasion/density scenario #####
scenario = 3

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
timesteps = 2000
pulse_time = 200
num_hours_each_time = 8
dt = 60*60*num_hours_each_time*1  # len of timesteps 

n_lines = 10 # note: # timesteps must be divisible by n_lines with timesteps%n_lines == 0
color = iter(plt.cm.viridis(np.linspace(0, 1, n_lines+1)))
c = next(color)

# bookkeeping 
new_dir_name = "Impact_of_Abrasion " + model_state ## Use when testing
#new_dir_name = "Longer_Run_Test_Scenario_1" + model_state

new_dir = pathlib.Path(os.getcwd(), new_dir_name)
new_dir.mkdir(parents=True, exist_ok=True)
output_folder = os.path.join(os.getcwd (), ("IOA_Results-"+ new_dir_name))
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


index_sort_back_to_landlab = np.zeros_like(index_sorted_area_link)
index_sort_back_to_landlab[index_sorted_area_link] = np.arange(grid.number_of_links)

#Sorted Drainage Area
area_link_DS = area_link[index_sorted_area_link]
area_node_DS = area_node[index_sorted_area_node]


increase_in_area = np.empty_like(area_link_DS)
increase_in_area[0] = area_link_DS[0]
increase_in_area[1:] = area_link_DS[1:] - area_link_DS[:-1]
print("Area link is", area_link_DS)
print("Increase in area is ", increase_in_area)

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

dist_downstream_DS = np.cumsum(length_DS)/1000

#Sorted Slope
slope = grid.at_link["channel_slope"]
slope_DS = slope[index_sorted_area_link]
initial_slope_DS= slope_DS.copy()

grid.at_link["dist_downstream"]= dist_downstream_DS[index_sort_back_to_landlab]

dist_downstream_nodes_DS = np.insert(dist_downstream_DS, 0, 0.0)

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

# creating sediment parcels in the DATARECORD
tau_c_multiplier = 2.4

initialize_parcels = BedParcelInitializerDepth(
    grid,
    flow_depth_at_link = depth,
    tau_c_50 = 0.15* slope**0.25, # slope dependent critical shields stress
    tau_c_multiplier = tau_c_multiplier,
    median_number_of_starting_parcels = 500,
    extra_parcel_attributes = ["source", "recycle_destination"]
    )

parcels = initialize_parcels()

parcels.dataset["source"].values = np.full(parcels.number_of_items, "initial_bed_sed")
#will be used to track the parcels that are recycled
parcels.dataset["recycle_destination"].values = np.full(parcels.number_of_items, "not_recycled_ever")

# calculation of the initial volume of sediment on each link

initial_vol_on_link = np.empty(number_of_links, dtype = float)
initial_vol_on_link= aggregate_items_as_sum(
    parcels.dataset.element_id.values[:, -1].astype(int),
    parcels.dataset.volume.values[:, -1],
    number_of_links,
)

D50_each_link = parcels.calc_aggregate_value(
    xr.Dataset.median, "D", at="link", fill_value=0.0)


# %% Plots related to grid... 

# Longitudinal Profile
plt.figure()
plt.plot(dist_downstream_DS, slope_DS, '-' )
plt.title("Starting channel Slope")
plt.ylabel("Slope")
plt.xlabel("Distance downstream (km)")
plt.show()

#FLOW DEPTH
plt.figure()
plt.plot(dist_downstream_DS, depth_DS, '-' )
plt.title("Flow Depth")
plt.ylabel("Flow depth (m)")
plt.xlabel("Distance downstream (km)")
plt.show()

# #Channel Width
plt.figure()
plt.plot(dist_downstream_DS, width_DS, '-' )
plt.title("Channel Width")
plt.ylabel("Channel Width (m)")
plt.xlabel("Distance downstream (km)")
plt.show()

tau_star = tau/((rho_sed - rho_water) * gravity * D50_each_link)

# Plot for shield stress
plt.figure()
plt.title("Tau_star")
plt.plot(dist_downstream_DS, tau_star)
plt.ylabel("Shield Stress")
plt.xlabel("Distance downstream (km)")
plt.show()

# D50 for each link
#find line of best fit
a, b = np.polyfit(dist_downstream_DS, D50_each_link[index_sorted_area_link], 1)
str_tau_c_multiplier = str(tau_c_multiplier)
plt.figure()
plt.scatter(dist_downstream_DS, D50_each_link[index_sorted_area_link])
plt.plot(dist_downstream_DS, a*dist_downstream_DS+b)  
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
    transport_method="WilcockCroweD50",
    active_layer_method = "Constant10cm",
    k_transp_dep_abr= 15.0,
)

# %% Setup for plotting during/after loop

#variables needed for plots

# Tracked for each timestep
n_recycled_parcels = np.ones(timesteps)*np.nan
vol_pulse_exited = np.ones(timesteps)*np.nan
vol_pulse_on = np.ones(timesteps)*np.nan
vol_pulse_abraded = np.ones(timesteps)*np.nan
percent_pulse_added = np.ones(timesteps)*np.nan
d_recycled_parcels = np.ones(timesteps)*np.nan
timestep_array = np.arange(timesteps)

# Tracked for each link, for each timestep
sed_total_vol = np.ones([grid.number_of_links,timesteps])*np.nan
sediment_active_percent = np.ones([grid.number_of_links,timesteps])*np.nan
D_mean_pulse_each_link= np.ones([grid.number_of_links,timesteps])*np.nan
D50s_each_link= np.ones([grid.number_of_links,timesteps])*np.nan
D_mean_active_each_link= np.ones([grid.number_of_links,timesteps])*np.nan
Change_D_each_link= np.zeros([grid.number_of_links,timesteps])
num_pulse_each_link= np.ones([grid.number_of_links,timesteps], dtype=int)*np.nan
num_each_link= np.ones([grid.number_of_links,timesteps], dtype=int)*np.nan
num_total_parcels_each_link= np.ones([grid.number_of_links,timesteps], dtype=int)*np.nan
transport_capacity= np.ones([grid.number_of_links,timesteps])*np.nan

num_active_parcels_each_link = np.ones([grid.number_of_links,timesteps])*np.nan
volume_at_each_link = np.ones([grid.number_of_links,timesteps])*np.nan
volume_pulse_at_each_link = np.ones([grid.number_of_links,timesteps])*np.nan
num_active_pulse= np.ones([grid.number_of_links,timesteps])*np.nan

time_array= np.arange(1, timesteps + 1)
days = (time_array * num_hours_each_time) / 24
num_days = days[-1]
time_array_ss= np.arange(pulse_time, timesteps+1)#ss = steady state
canyon_reaches= index_sorted_area_link[43:54]
Sed_flux= np.empty([grid.number_of_links, timesteps])
Slope_through_time_DS = np.empty([grid.number_of_links, timesteps])
Sand_fraction_active = np.ones([grid.number_of_links,timesteps])*np.nan

Elev_change_DS = np.empty([grid.number_of_nodes,timesteps]) # 2d array of elevation change in each timestep 


# %% Pulse element variables
#determine number of pulse parcels
fan_thickness = np.array([3.0, 6.25, 6.25, 5.75, 2.75, 1.25]) #meters
fan_location= np.array([6, 7, 8, 9, 10, 11])
pulse_parcel_vol = 25 #volume of each parcel m^3 - edited 1/9/25
total_num_pulse_parcels = np.int64(4900000/pulse_parcel_vol * (1-bed_porosity))  


initial_pulse_volume = fan_thickness*length[fan_location]*width[fan_location] # m^3; volume of pulse that should be added to each link of the Chocolate Fan
initial_pulse_rock_volume = initial_pulse_volume * (1-bed_porosity)
initial_num_pulse_parcels_by_vol = initial_pulse_rock_volume/pulse_parcel_vol #number of parcels added to each link based on volume; 
total_num_initial_pulse_parcels = int(np.sum(initial_num_pulse_parcels_by_vol))

total_num_added_pulse_parcels = 0

## distribute the total_num_pulse_parcels proportional to fan thickness instead of evenly
fan_proportions = fan_thickness / fan_thickness.sum()
# Distribute total_num_pulse_parcels according to the proportions
pulse_each_fan_link = (fan_proportions * total_num_pulse_parcels).astype(int) #
# Calculate the remainder due to integer conversion
remainding_pulse = total_num_pulse_parcels - pulse_each_fan_link.sum()
pulse_each_fan_link[np.argmax(fan_proportions)] += remainding_pulse


#pulse variables

pulse_location = [index for index, freq in enumerate((pulse_each_fan_link).astype(int), start= np.min(fan_location)) for _ in range(freq)]

random.shuffle(pulse_location)

newpar_element_id = pulse_location
newpar_element_id = np.expand_dims(newpar_element_id, axis=1)   


new_starting_link = np.squeeze(pulse_location)

np.random.seed(0)
 
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

field_grain_sizes = pd.read_excel((os.path.join(os.getcwd (), ("SuiattleFieldData_Combined20182019.xlsx"))), sheet_name = '3 - Grain size df deposit', header = 0)
Suiattle_gsd = field_grain_sizes["Size (cm)"].values
Suiattle_gsd_m = Suiattle_gsd/100
scaled_size = np.int64(total_num_pulse_parcels * 10)

# Scale up by repeating the array
scaled_gsd = np.resize(Suiattle_gsd_m, scaled_size)
sorted_scaled_gsd = np.sort(scaled_gsd)
scaled_count = np.linspace(0,100,len(scaled_gsd))

new_D = np.empty(((np.shape(newpar_element_id)[0]), 1), dtype=float)

# Filter values in GSD that are less than 0.5
truncated_GSD = scaled_gsd[scaled_gsd < 0.5]

new_D[:,0] = truncated_GSD[:total_num_pulse_parcels]

count = np.linspace(0,100,len(Suiattle_gsd_m))

# Plot of orginal GSD and scaled GSD (plots overlap)
plt.semilogx(np.sort(Suiattle_gsd_m),count, color ="blue")
plt.semilogx(np.sort(scaled_gsd),scaled_count, color ="blue")
plt.xlabel('Grain Size (log scale)')
plt.ylabel('Cumulative % Finer')
plt.show()

# %% Abrasion scenarios using Pfeiffer et al., 2022 field data
field_data = pd.read_excel((os.path.join(os.getcwd (), ("SuiattleFieldData_Combined20182019.xlsx"))), sheet_name = '4-SHRSdfDeposit', header = 0)
SHRS = field_data['SHRS median'].values
SHRS_MEAN = np.mean(SHRS)
SHRS_STDEV = np.std(SHRS)

SHRS_distribution = np.random.normal(SHRS_MEAN, SHRS_STDEV, (np.size(newpar_element_id)))

SHRS_distribution[SHRS_distribution<20]=20  # Hard coded to prevent occasional VERY weak values. Min set to just below observed min (SHRS = 24)

measured_alphas = np.exp(1.122150830896329)*np.exp(-0.13637476779600016 *SHRS_distribution) # units: 1/km
 
# tumbler correction factors
tumbler_2 = 2 * measured_alphas
tumbler_4 = 4 * measured_alphas


if scenario == 1:
    new_abrasion_rate = 0 * np.ones(np.size(newpar_element_id))
    new_density = 2650 * np.ones(np.size(newpar_element_id))  # (kg/m3) standard rock density
    print('Scenario 1 pulse: no abrasion')
elif scenario == 2:
    new_abrasion_rate = (measured_alphas/1000)* np.ones(np.size(newpar_element_id)) 
    new_density = 928.6259337721524*np.log(SHRS_distribution) + (-1288.1880919573282)
    print('Scenario 2 pulse: variable abrasion, SH proxy')
else: 
    new_abrasion_rate = (tumbler_4/1000)* np.ones(np.size(newpar_element_id)) 
    new_density = 928.6259337721524*np.log(SHRS_distribution) + (-1288.1880919573282)
    print('Scenario 3 pulse: variable abrasion, 4 * SH proxy')
    

flat_new_D = new_D.reshape(-1)
new_abrasion_rate[flat_new_D < 0.002] = 0

newer_pulse_D_by_timestep = {}

# %% Animation start
# Initiate an animation writer using the matplotlib module, `animation`.
figAnim, axAnim = plt.subplots(1, 1)
#writer = animation.FFMpegWriter(fps=6)

Writer = animation.writers['pillow']
writer = Writer(fps=6)

gif_name = output_folder +"/"+ new_dir_name + "--Elev_through_time.gif"
writer.setup(figAnim,gif_name)

# %% Model run

for t in range(0, (timesteps*dt), dt):
    start_time = model_time.time()
    
    day_timestep = t/(60*60*24)
    if day_timestep % 20 == 0:
        discharge = discharge * 2.5
        print("0.1% exceedence flow for discharge")
    
    else:
        discharge = (0.2477 * area_link) + 8.8786
    
# %%    # ###########   Parcel recycling by drainage area    ###########    
    
    # Masking parcels
    
    #boolean mask of parcels that have left the network
    mask1 = parcels.dataset.element_id.values[:,-1] == OUT_OF_NETWORK
    #boolean mask of parcels that are initial bed parcels (no pulse parcels)
    mask2 = parcels.dataset.source == "initial_bed_sed" 
    
    
    bed_sed_out_network = parcels.dataset.element_id.values[mask1 & mask2, -1] == -2
    
    
    #index of the bed parcels that have left the network
    index_exiting_parcels = np.where(bed_sed_out_network == True) 
    print("Number of recycling parcels is: ", np.size(index_exiting_parcels))
    
    #number of parcels that will be recycled
    num_exiting_parcels = np.size(index_exiting_parcels)
    
    # Calculate the total sum of area_link
    
    total_increase_area = sum(increase_in_area)

    # Calculate the proportions based on drainage area
    proportions = increase_in_area/total_increase_area

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
    
    n_recycled_parcels[np.int64(t/dt)]=np.sum(parcels.dataset.element_id.values[:,-1] == OUT_OF_NETWORK,-1)
    d_recycled_parcels[np.int64(t/dt)]=np.mean(parcels.dataset.D.values[parcels.dataset.element_id.values[:,-1] == OUT_OF_NETWORK,-1])
        
        # assign the new starting link
    if np.size(bed_sed_out_network) > 0:    
        parcels.dataset.element_id.values[index_exiting_parcels, -1] = indices_recyc_bed
        parcels.dataset.starting_link.values[index_exiting_parcels] = OUT_OF_NETWORK # way to denote it was recycled
        parcels.dataset.location_in_link.values[index_exiting_parcels] = 0.
        
        parcels.dataset["recycle_destination"].values[index_exiting_parcels] = np.full(num_exiting_parcels, f"recycle_des {indices_recyc_bed}")
        
        
    else:
         print("There are no recycled parcels in this timestep.")
# %% ########## Making a pulse #############  
    mask_ispulse = parcels.dataset.source == 'pulse'
   
    # Calculate the current timestep
    current_timestep = t // dt
    
    if total_num_pulse_parcels > total_num_added_pulse_parcels:
        

    # Check if it's the time for pulse evaluation
        if current_timestep >= pulse_time and (current_timestep - pulse_time) % 5 == 0: 
    
            if t == dt * pulse_time:  
                
                print ("# First pulse time")
            
                new_time_arrival_in_link = nst._time_idx + np.expand_dims(
                        np.random.uniform(size=total_num_initial_pulse_parcels), axis=1
                    )

                new_parcels = {"grid_element": newpar_grid_elements[:total_num_initial_pulse_parcels],
                      "element_id": newpar_element_id[:total_num_initial_pulse_parcels]}

                new_variables = {
                    "starting_link": (["item_id"], new_starting_link[:total_num_initial_pulse_parcels]),
                    "abrasion_rate": (["item_id"], new_abrasion_rate[:total_num_initial_pulse_parcels]),
                    "density": (["item_id"], new_density[:total_num_initial_pulse_parcels]),
                    "source": (["item_id"], new_source[:total_num_initial_pulse_parcels]),
                    "time_arrival_in_link": (["item_id", "time"], new_time_arrival_in_link),
                    "active_layer": (["item_id", "time"], new_active_layer[:total_num_initial_pulse_parcels]),
                    "location_in_link": (["item_id", "time"], new_location_in_link[:total_num_initial_pulse_parcels]),
                    "D": (["item_id", "time"], new_D[:total_num_initial_pulse_parcels]),
                    "volume": (["item_id", "time"], new_volume[:total_num_initial_pulse_parcels]),
                    }
                
                parcels.add_item(
                        time=[nst._time],
                        new_item = new_parcels,
                        new_item_spec = new_variables
                        )
                
                total_num_added_pulse_parcels = total_num_initial_pulse_parcels
                D_pulse_initial= new_D[:total_num_initial_pulse_parcels]

            else:

                elevation_above_bedrock = grid.at_node['topographic__elevation'][:-1]-grid.at_node['bedrock__elevation'][:-1]   
                height_to_add_fan_links = fan_thickness- elevation_above_bedrock[fan_location]
                
                height_to_add_fan_links[height_to_add_fan_links<0]=0
                height_to_add_fan_links = np.nan_to_num(height_to_add_fan_links, nan=0)
                vol_to_add_fan_links = height_to_add_fan_links*length[fan_location]*width[fan_location]
                num_parcels_add_to_fan_now = np.int64(vol_to_add_fan_links/pulse_parcel_vol)
                
                num_new_pulse = np.sum(num_parcels_add_to_fan_now)
                
                # Loop through pulse parcels to check their location
                new_pulse_location = [index for index, freq in enumerate((num_parcels_add_to_fan_now).astype(int), start= np.min(fan_location)) for _ in range(freq)]
                
                print("The number of pulse added to each link is ", num_parcels_add_to_fan_now)
                
                new_starting_link = np.squeeze(new_pulse_location) 
                
                if num_new_pulse >= 2: 
                    
                    newpar_element_id = new_pulse_location
                    newpar_element_id = np.expand_dims(newpar_element_id, axis=1)   
                    
                    new_time_arrival_in_link = nst._time_idx + np.expand_dims(
                            np.random.uniform(size=num_new_pulse), axis=1
                        )

                    print('The total number of new pulse parcels added =', num_new_pulse)
            
                    new_parcels = {"grid_element": newpar_grid_elements[total_num_initial_pulse_parcels:total_num_initial_pulse_parcels+num_new_pulse],
                            "element_id": newpar_element_id}

                    new_variables = {
                    "starting_link": (["item_id"], new_starting_link),
                    "abrasion_rate": (["item_id"], new_abrasion_rate[total_num_initial_pulse_parcels:total_num_initial_pulse_parcels+num_new_pulse]),
                    "density": (["item_id"], new_density[total_num_initial_pulse_parcels:total_num_initial_pulse_parcels+num_new_pulse]),
                    "source": (["item_id"], new_source[total_num_initial_pulse_parcels:total_num_initial_pulse_parcels+num_new_pulse]),
                    "time_arrival_in_link": (["item_id", "time"], new_time_arrival_in_link),
                    "active_layer": (["item_id", "time"], new_active_layer[total_num_initial_pulse_parcels:total_num_initial_pulse_parcels+num_new_pulse]),
                    "location_in_link": (["item_id", "time"], new_location_in_link[total_num_initial_pulse_parcels:total_num_initial_pulse_parcels+num_new_pulse]),
                    "D": (["item_id", "time"], new_D[total_num_initial_pulse_parcels:total_num_initial_pulse_parcels+num_new_pulse]),
                    "volume": (["item_id", "time"], new_volume[total_num_initial_pulse_parcels:total_num_initial_pulse_parcels+num_new_pulse]),
                      
                        }

                    total_num_added_pulse_parcels += num_new_pulse
                    newer_pulse_D_by_timestep[f"newer_pulse_D_timestep_{current_timestep}"] = new_D[total_num_initial_pulse_parcels:total_num_initial_pulse_parcels+num_new_pulse]
                    print('The number of new parcels in this timestep is =', num_new_pulse)
                    
                parcels.add_item(
                            time=[nst._time],
                            new_item = new_parcels,
                            new_item_spec = new_variables
                            )
                   

            percent_pulse_added[np.int64(t/dt)] = total_num_added_pulse_parcels/total_num_pulse_parcels
            
            print('making a pulse')
            print("On this timestep the total number of pulse parcels added = ", total_num_added_pulse_parcels)
            
            
            print('t = ',t, ',timestep = ', t/dt) 
            
            
    else:
        continue
# %% Run one timestep    
    
    nst.run_one_step(dt)

    
#%% #Calculate Variables and tracking timesteps
    if t == parcels.dataset.time[0].values: # if we're at the first timestep

        avg_init_sed_thickness = np.mean(
            grid.at_node['topographic__elevation'][:-1]-grid.at_node['bedrock__elevation'][:-1])        
        grid.at_node['bedrock__elevation'][-1]=grid.at_node['bedrock__elevation'][-1]+avg_init_sed_thickness


        elev_initial = grid.at_node['topographic__elevation'].copy()
        
        
    Elev_change_DS[:,np.int64(t/dt)] = (grid.at_node['topographic__elevation']-elev_initial)[index_sorted_area_node]
    

        
    if t%((timesteps*dt)/n_lines)==0:  
        
        # Animation
        plt.figure(figAnim)
        
    
        elev_change = grid.at_node['topographic__elevation']-elev_initial
        
        plt.plot(dist_downstream_nodes_DS, elev_change[index_sorted_area_node], c= "darkblue")
        plt.xlabel('Distance downstream (km)')
        plt.title("Elevation change through time")
        plt.ylabel('Elevation change (m)')
        plt.ylim(-7, 18)
        text = plt.text(35,7,str(np.round(t/(60*60*24),1))+" Days")
      
        writer.grab_frame()
    
        # shade it out for next frame, clear text
        text.remove()
        plt.plot(dist_downstream_nodes_DS, elev_change[index_sorted_area_node], c = 'w', alpha = 0.7)
        plt.pause(0.001)
    
    

    #Tracking timesteps
    print("Model time: ", t/(60*60*24), " * 8 hours passed (", t/dt, 'timesteps)')
    print('Elapsed:', (model_time.time() - start_time)/60 ,' minutes')
    
    # Print some pertinent information about the model run
    print('mean grain size of pulse',np.mean(
        parcels.dataset.D.values[parcels.dataset.source.values[:] == 'pulse',-1]))
    
# %% Calculating transport capacity and volume remaining after abrasion
    #Additional masks
    mask_ispulse = parcels.dataset.source == 'pulse'
    mask_active = parcels.dataset.active_layer[:,-1]==1 
    
    #Arrays of elements_ids of various 
    element_all_parcels = parcels.dataset.element_id.values[:, -1].astype(int)
    element_pulse = element_all_parcels[mask_ispulse]
    element_active = element_all_parcels[mask_active]

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
    
    Slope_through_time_DS[:,np.int64(t/dt)] = grid.at_link["channel_slope"][index_sorted_area_link] - initial_slope_DS
    
    Sand_fraction_active[:,np.int64(t/dt)] = grid.at_link["sediment__active__sand_fraction"]
    
    # Calculating volume of pulse that remains after abrasion    
    volume_of_pulse = parcels.dataset.volume.values[mask_ispulse,-1]

    vol_pulse_on[np.int64(t/dt)]=np.sum(volume_of_pulse[element_pulse!= OUT_OF_NETWORK])
    vol_pulse_exited[np.int64(t/dt)]=np.sum(volume_of_pulse[element_pulse == OUT_OF_NETWORK])
    
    vol_pulse_abraded[np.int64(t/dt)]= ((total_num_added_pulse_parcels
                                        * pulse_parcel_vol)
                                        - vol_pulse_on[np.int64(t/dt)]
                                        - vol_pulse_exited[np.int64(t/dt)]
                                        )     
    
    
#%%  Aggregators for all parcels
    
    # Populating arrays with data from this timestep. 
    sed_act_vol= grid.at_link["sediment__active__volume"][index_sorted_area_link] 
  
    sed_total_vol[:,np.int64(t/dt)] = grid.at_link["sediment_total_volume"][index_sorted_area_link]
    
    # Create a mask where sed_total_vol is non-zero
    mask_sed_vol = sed_total_vol[:, np.int64(t/dt)] != 0

    # Initialize sediment_active_percent with zeros (to handle the cases where both arrays are zero)
    sediment_active_percent[:, np.int64(t/dt)] = 0

    # Perform division only where the mask is True (i.e., sed_total_vol is not zero)
    sediment_active_percent[mask_sed_vol, np.int64(t/dt)] = sed_act_vol[mask_sed_vol] / sed_total_vol[mask_sed_vol, np.int64(t/dt)]
    
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
    
    D_mean_active_each_link[:,np.int64(t/dt)] = aggregate_items_as_mean(
        element_active,
        parcels.dataset.D.values[:,-1],
        parcels.dataset.volume.values[:, -1],
        number_of_links,
        )
    
    D50s_each_link[:,np.int64(t/dt)] = nst._d50_active.copy() # THIS DOESN"T WORK.. why?
    
    num_total_parcels_each_link[:, np.int64(t/dt)] = aggregate_items_as_count(
        element_all_parcels,
        number_of_links,
        )
    
    num_each_link[:, np.int64(t/dt)] = aggregate_items_as_count(
        element_all_parcels,
        number_of_links,
        )


    if t > dt*pulse_time :

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
            
plt.figure(figAnim)
plt.close()

writer.finish()
    
# %% -----> END MODEL RUN <----   
    

# %% Variable bookkeeping - downstream sorting, etc

pre_pulse_D50s = np.mean(D50s_each_link[:, pulse_time-20:pulse_time-1],axis = 1) # average of 10 timesteps before
pre_pulse_D50s = pre_pulse_D50s.reshape(-1, 1)

Change_D50s_each_link = D50s_each_link - pre_pulse_D50s

pre_pulse_Dmean = D_mean_active_each_link[:, pulse_time-1].reshape(-1, 1)  # Reshape to (number_of_links, 1) for broadcasting
Change_Dmean_each_link = D_mean_active_each_link - pre_pulse_Dmean

num_active_parcels_each_link_DS = num_active_parcels_each_link[index_sorted_area_link]
volume_at_each_link_DS = volume_at_each_link[index_sorted_area_link]
D50s_each_link_DS = D50s_each_link[index_sorted_area_link]
Change_D50s_each_link_DS = Change_D50s_each_link[index_sorted_area_link]
Change_Dmean_each_link_DS = Change_Dmean_each_link[index_sorted_area_link]
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

# %% variables for plots below

mask_active_pulse = np.nan_to_num(num_pulse_each_link, nan=0.0) != 0 # Replace NaNs with zeros and create a mask where the denominator is non-zero

# Perform the division where valid, otherwise set to zero
percent_active_pulse = np.divide(num_active_pulse, num_pulse_each_link, where=mask_active_pulse, out=np.zeros_like(num_active_pulse, dtype=float))

percent_active_pulse[np.isnan(num_pulse_each_link)] = np.nan

percent_active_pulse_DS = percent_active_pulse[index_sorted_area_link] 
percent_active_pulse_DS[percent_active_pulse_DS == 0]= np.nan

mask_active_parcels = num_each_link != 0 # Create a mask where the denominator is non-zero

percent_active = np.zeros_like(num_active_parcels_each_link, dtype=float)

percent_active[mask_active_parcels] = num_active_parcels_each_link[mask_active_parcels] / num_each_link[mask_active_parcels]# Perform division only where num_each_link is non-zero
percent_active_DS = percent_active[index_sorted_area_link] 
percent_active_DS[percent_active_DS == 0]= np.nan

# final volume of sediment on each link
final_vol_on_link = np.empty(number_of_links, dtype = float)
final_vol_on_link= aggregate_items_as_sum(
    element_all_parcels,
    parcels.dataset.volume.values[:, -1],
    number_of_links,
)
Final_vol_on_link_DS = final_vol_on_link[index_sorted_area_link] 


## Adding elevation change as an at_link variable
elev_change_at_link = ((final_vol_on_link - initial_vol_on_link) /
                        (grid.at_link["reach_length"]*grid.at_link["channel_width"]))  # volume change in link/ link bed area


# %% All the Pcolor plots

# Misc ones
plt.figure(dpi=600)
plt.pcolor(days, dist_downstream_DS, Sand_fraction_active [index_sorted_area_link] , 
            shading='nearest', 
            norm= None, 
            cmap="copper"
            ) 
plt.colorbar(label= "Sand fraction")
plt.xlabel('Days')
plt.ylabel('Distance downstream (km)')
plt.axvline(x=pulse_time*(dt/(60*60*24)), color='gray', linestyle='--', linewidth=1.5, label=f'Day {pulse_time}')
plot_name = str(new_dir) + "/" + 'Sand_frac(Scenario' + str(scenario_num) + ').png'
plt.savefig(plot_name)
plt.show()

plt.figure(dpi=600)
plt.plot(days, percent_pulse_added, '.') 
plt.xlabel('Days')
plt.ylabel('Fraction of Pulse added')
plot_name = str(new_dir) + "/" + 'percentage of the pulse added (Scenario' + str(scenario_num) + ').png'
plt.savefig(plot_name)
plt.show()

plt.figure(dpi=600)
plt.pcolor(days, dist_downstream_DS, Slope_through_time_DS, cmap = 'RdBu', norm = colors.CenteredNorm())
plt.colorbar(label= "Change in Slope")
plt.xlabel('Days')
plt.axvline(x=pulse_time*(dt/(60*60*24)), color='gray', linestyle='--', linewidth=1.5, label=f'Day {pulse_time}')
plt.ylabel("Distance downstream (km)")
plot_name = str(new_dir) + "/" + 'slope_change (Scenario' + str(scenario_num) + ').png'
plt.savefig(plot_name)
plt.show()

plt.figure(dpi=600)
plt.pcolor(days, dist_downstream_DS, Change_Dmean_each_link_DS, cmap = 'PuOr', norm = colors.CenteredNorm() )
plt.colorbar(label= "Mean surface D change (m)")
plt.xlabel('Days')
plt.ylabel("Distance downstream (km)")
plot_name = str(new_dir) + "/" + 'Dmean_change (Scenario' + str(scenario_num) + ').png'
plt.savefig(plot_name)
plt.show()

plt.figure(dpi=600)
plt.pcolor(days, dist_downstream_DS, Change_D50s_each_link_DS, cmap = 'PuOr', norm = colors.CenteredNorm(halfrange=0.3) )
plt.colorbar(label= "D50s change (m)")
plt.xlabel('Days')
plt.axvline(x=pulse_time*(dt/(60*60*24)), color='gray', linestyle='--', linewidth=1.5, label=f'Day {pulse_time}')
plt.ylabel("Distance downstream (km)")
plot_name = str(new_dir) + "/" + 'D50s_change (Scenario' + str(scenario_num) + ').png'
plt.savefig(plot_name)
plt.show()


plt.figure(dpi=600)
plt.pcolor(days, dist_downstream_nodes_DS, Elev_change_DS, cmap = 'RdBu', norm = colors.CenteredNorm(halfrange=1.5) )
plt.colorbar(label= "Elevation change from initial (m)")
plt.xlabel('Days')
plt.axvline(x=pulse_time*(dt/(60*60*24)), color='gray', linestyle='--', linewidth=1.5, label=f'Day {pulse_time}')
plt.ylabel("Distance downstream (km)")
plot_name = str(new_dir) + "/" + 'ELevhalfrange_change (Scenario' + str(scenario_num) + ').png'
plt.savefig(plot_name)
plt.show()

# Many more
variables = {
    'sediment_active_percent': sediment_active_percent,
    'num_pulse_each_link_DS': num_pulse_each_link_DS,
    'num_total_parcels_each_link_DS': num_total_parcels_each_link_DS,
    'num_active_pulse_nozero_DS': num_active_pulse_nozero_DS,
    'vol_pulse_nozero_DS': vol_pulse_nozero_DS,
    'vol_nozero_DS': vol_nozero_DS,
    'D_mean_pulse_each_link_nozero_DS': D_mean_pulse_each_link_nozero_DS,
    'percent_active_pulse_DS': percent_active_pulse_DS,
    'percent_active_DS': percent_active_DS,
    'Elev_change_DS': Elev_change_DS,
}
# Define colorbar labels with new names
colorbar_labels = {
    'sediment_active_percent': 'Average % active parcels',
    'num_pulse_each_link_DS': 'Number of pulse parcel',
    'num_total_parcels_each_link_DS': 'Number of parcels',
    'num_active_pulse_nozero_DS': 'Number of active pulse parcels',
    'vol_pulse_nozero_DS': 'Volume of pulse parcels ($m^3$)',
    'vol_nozero_DS': 'Volume of all parcels ($m^3$)',
    'D_mean_pulse_each_link_nozero_DS': 'Mean grain size of pulse parcels (m)',
    'percent_active_pulse_DS': 'Fraction of active pulse parcels',
    'percent_active_DS': 'Fraction of active parcels',
    'Elev_change_DS': 'Elevation change from initial (m)',

}

plot_titles = {
    'sediment_active_percent': f"Active parcel volume (Scenario = {scenario})",
    'num_pulse_each_link_DS': f"Number of total pulse parcels (Scenario = {scenario})",
    'num_total_parcels_each_link_DS': f"Number of total parcels (Scenario = {scenario})",
    'num_active_pulse_nozero_DS': f"Number of active pulse parcels (Scenario = {scenario})",
    'vol_pulse_nozero_DS': f"Volume of pulse parcel (Scenario = {scenario})",
    'vol_nozero_DS': f"Volume of parcel (Scenario = {scenario})",
    'D_mean_pulse_each_link_nozero_DS': f"Dmean pulse grain size (Scenario = {scenario})",
    'percent_active_pulse_DS': f"Fraction of active pulse parcels (Scenario = {scenario})",
    'percent_active_DS': f"Fraction of  active parcels (Scenario = {scenario})",
    'Elev_change_DS': f"Elevation change (Scenario = {scenario})",
}
plot_names = {
    'sediment_active_percent': f"{new_dir}/Percent_Active (Scenario{scenario_num}).png",
    'num_pulse_each_link_DS': f"{new_dir}/TotalPulseParcel(Scenario{scenario_num}).png",
    'num_total_parcels_each_link_DS': f"{new_dir}/Total_parcel(Scenario{scenario_num}).png",
    'num_active_pulse_nozero_DS': f"{new_dir}/ActivePulse(Scenario{scenario_num}).png",
    'vol_pulse_nozero_DS': f"{new_dir}/Vol_pulse(Scenario{scenario_num}).png",
    'vol_nozero_DS': f"{new_dir}/Vol(Scenario{scenario_num}).png",
    'D_mean_pulse_each_link_nozero_DS': f"{new_dir}/D_mean_pulse(Scenario{scenario_num}).png",
    'percent_active_pulse_DS': f"{new_dir}/fraction_active_pulse(Scenario{scenario_num}).png",
    'percent_active_DS': f"{new_dir}/Fraction_Active (Scenario{scenario_num}).png",
    'Elev_change_DS': f"{new_dir}/TrackElevChange(Scenario{scenario_num}).png",
}
cmap = {
    'sediment_active_percent': 'viridis',
    'num_pulse_each_link_DS': 'inferno',
    'num_total_parcels_each_link_DS': 'inferno',
    'num_active_pulse_nozero_DS': 'winter_r',
    'vol_pulse_nozero_DS': 'plasma_r',
    'vol_nozero_DS': 'plasma_r',
    'D_mean_pulse_each_link_nozero_DS': 'winter_r',
    'percent_active_pulse_DS': 'Wistia',
    'percent_active_DS': 'Wistia',
    'Elev_change_DS': 'RdBu',
}

norm = {
    'sediment_active_percent': None,
    'num_pulse_each_link_DS': None,
    'num_total_parcels_each_link_DS': None,
    'num_active_pulse_nozero_DS': None,
    'vol_pulse_nozero_DS': matplotlib.colors.LogNorm(),
    'vol_nozero_DS': matplotlib.colors.LogNorm(),
    'D_mean_pulse_each_link_nozero_DS': matplotlib.colors.LogNorm(vmin=0.005),
    'percent_active_pulse_DS': matplotlib.colors.Normalize(vmin=0.0, vmax=1.0),
    'percent_active_DS': matplotlib.colors.Normalize(vmin=0.0, vmax=1.0),
    'Elev_change_DS': colors.CenteredNorm(),
}

# Plot for all time
for var_name, data in variables.items():
    plt.figure()
    
    if var_name == 'Elev_change_DS':  # Special handling for Elev_change
        mesh = plt.pcolor(days, dist_downstream_nodes_DS, data, shading='auto', cmap=cmap[var_name], norm=norm[var_name])

    else:
        mesh = plt.pcolor(days, dist_downstream_DS, data, shading='auto', cmap=cmap[var_name], norm=norm[var_name] if norm[var_name] is not None else None)
        
    x_right_edge = plt.gca().get_xlim()[1]
    pulseline = plt.plot(np.ones(len(fan_location))*x_right_edge*1.005,dist_downstream_DS[fan_location],color='red')
    pulseline[0].set_clip_on(False)
    
    plt.text(x_right_edge*1.01, 7, "Pulse", rotation= 90, va= 'center', ha= 'left', color='red')
    plt.axis([days[0],num_days,0,max(dist_downstream_nodes_DS)])
    plt.axvline(x=pulse_time*(dt/(60*60*24)), color='gray', linestyle='--', linewidth=1.5, label=f'Day {pulse_time}')
   
    mesh.set_array(data)
    plt.colorbar(mesh, label=colorbar_labels[var_name])
    plt.title(plot_titles[var_name])
    plt.xlabel("Days")
    plt.ylabel("Distance downstream (km)")
    plt.savefig(plot_names[var_name], dpi= 700)
    plt.show()    


# %% Misc other plots
# #### Total pulse volume through time (constant for no-abrasion, until parcels exit)
plt.figure(dpi=600,figsize=(6,2))

plt.stackplot(days,vol_pulse_on,vol_pulse_exited,vol_pulse_abraded, colors = ['k','darkorange','darkgrey'])

plt.ylim(0,np.nanmax(vol_pulse_on + vol_pulse_exited + vol_pulse_abraded))
plt.xlim(0,np.max(days)) 
median_y= np.nanmedian(vol_pulse_on + vol_pulse_exited + vol_pulse_abraded)

#plt.text(np.max(days)*0.6,np.median(vol_pulse_on)*0.5,'(pulse sed in channel)', color = 'white')

if np.max(vol_pulse_exited)>0.05*median_y:
    vloc = np.median(vol_pulse_on) + 0.9 * (np.median(vol_pulse_exited))
    #plt.text(np.max(days)*0.6,vloc,'(exited channel)', color = 'white')
    
if np.max(vol_pulse_abraded)>0.05*median_y:
    vloc = median_y +0.9*(np.nanmedian(vol_pulse_abraded))
    #plt.text(np.max(days)*0.6,vloc,'(lost to abrasion)', color = 'black')
    
plt.xlabel('Days')
plt.ylabel(r'Total pulse volume (m$^3$)')

plot_name = str(new_dir) + "/" + 'PulseAbrasionRemaining(Scenario' + str(scenario_num) + ').png'
plt.savefig(plot_name, dpi=700)


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
ax1.plot(days, d_recycled_parcels,'-', color='brown')
ax2.plot(days, n_recycled_parcels,'-', color='k') #,'.' makes it a dot chart, '-' makes line chart
ax1.set_xlabel("Days")
ax1.set_ylabel("Mean D recycled parcels (m)", color='brown')
plt.title("Recycled parcels (Scenario = %i)" %scenario)
ax2.set_ylabel("Total number of recycled parcels", color='k')
plot_name = str(new_dir) + "/" + 'Recycled(Scenario' + str(scenario_num) + ').png'
plt.savefig(plot_name, dpi=700)
plt.show()

# How far have the non-pulse parcels traveled in total?
travel_dist = nst._distance_traveled_cumulative[:-total_num_initial_pulse_parcels]
nonzero_travel_dist = travel_dist[travel_dist>0]
plt.hist(np.log10(nonzero_travel_dist),30) # better to make log-spaced bins...
plt.xlabel('log10( Cum. parcel travel distance (m) )')
plt.ylabel('Number of non-pulse parcels')
plt.title("How far have the non-pulse parcels traveled in total? (Scenario = %i)" %scenario)
plot_name = str(new_dir) + "/" + 'log10( Cum. parcel travel distance (m) (Scenario' + str(scenario_num) + ').png'
plt.savefig(plot_name, dpi=700)
plt.text(2,50, 'n = '+str(len(nonzero_travel_dist))+' parcels')
plt.text(2,50, 'n = '+str(len(nonzero_travel_dist))+' parcels')

# Cumulative D comparing pulse before and after transport

pulse_mask = parcels.dataset['source'].values == 'pulse'
D_values = parcels.dataset.D.values[pulse_mask, :]

filtered_nan_mask = ~np.isnan(D_values[:, pulse_time]) & ~np.isnan(D_values[:, -1])

initial_pulse_D = D_values[filtered_nan_mask, pulse_time]
final_pulse_D = D_values[filtered_nan_mask, -1]

sorted_initial_D = np.sort(initial_pulse_D)
sorted_final_D = np.sort(final_pulse_D)

cumulative_initial_pulse = np.linspace(0, 1, len(sorted_initial_D))
cumulative_final_pulse = np.linspace(0, 1, len(sorted_final_D))

plt.figure(figsize=(8, 6))
plt.plot(sorted_initial_D, cumulative_initial_pulse, label='Initial pulse GSD', color='blue')
plt.plot(sorted_final_D, cumulative_final_pulse, label='Final pulse GSD', color='red')
plt.xlabel('Pulse parcel grain size (m)')
plt.ylabel('Cumulative Probability')
plt.title('Initial and final pulse grain size CDF (source=pulse)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xscale('log')
plt.xlim(0.002,1)
plt.legend()
plot_name = str(new_dir) + "/" + 'CumFineGSD(Scenario' + str(scenario_num) + ').png'
plt.savefig(plot_name, dpi=700)
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
ax1.plot(dist_downstream_DS, D50_each_link[index_sorted_area_link],'-', color='k')
ax2.plot(dist_downstream_DS, num_total_parcels_each_link_DS,'-', color='brown') #,'.' makes it a dot chart, '-' makes line chart
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

plt.figure()
plt.plot(dist_downstream_DS, grid.at_link["channel_slope"][index_sorted_area_link], '-', color= 'red',  label = "Current slope" )
plt.plot(dist_downstream_DS, initial_slope_DS, '-', color ='black', label ="Initial slope" )
plt.title("Starting channel Slope")
plt.ylabel("Slope")
plt.xlabel("Distance downstream (km)")
plt.legend()
plt.show()

# %% GIF within a link parcels

for link in np.array([10,11,12,13,14,15,16,17, 18, 19, 20]):#range(16):#range(grid.number_of_links):
    output_folder = os.path.join(os.getcwd (), ("Gifs replace this folder"))
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # Initiate an animation writer using the matplotlib module, `animation`.
    figPulseAnim, axPulseAnim = plt.subplots(1, 1, dpi=300)
    #writer = animation.FFMpegWriter(fps=20)
    
    Writer = animation.writers['pillow']
    writer = Writer(fps=20)
    
    gif_name = output_folder +"/"+ scenario_num + "Test Wiggle" + str(link) +".gif"
    writer.setup(figPulseAnim,gif_name)
    for tstep in np.arange(0,timesteps,5):#range(timesteps):
    # for tstep in [500]: # for now, doing for just one timestep
        mask_here = parcels.dataset.element_id.values[:,tstep] == link
        time_arrival = parcels.dataset.time_arrival_in_link.values[mask_here, tstep]
        volumes = parcels.dataset.volume.values[:, tstep]
        current_link = parcels.dataset.element_id.values[:,tstep]#.astype(np.int64)
        this_links_parcels = np.where(current_link == link)[0]
        time_arrival_sort =np.argsort(time_arrival,0,)
        parcel_id_time_sorted = this_links_parcels[time_arrival_sort]
        vol_ordered_filo = volumes[parcel_id_time_sorted]
        cumvol_orderedfilo = np.cumsum(volumes[parcel_id_time_sorted])
        effectiveheight_orderedfilo = cumvol_orderedfilo/(grid.at_link["channel_width"][link]*grid.at_link["reach_length"][link])
        source_orderedfilo = parcels.dataset.source[parcel_id_time_sorted]
        active_orderedfilo = parcels.dataset.active_layer[parcel_id_time_sorted,tstep]
        location_in_link_orderedfilo = parcels.dataset.location_in_link.values[parcel_id_time_sorted,tstep]
        D_orderedfilo = parcels.dataset.D.values[parcel_id_time_sorted,tstep]
        # Plot
        plt.scatter(location_in_link_orderedfilo[active_orderedfilo==1],
                    effectiveheight_orderedfilo[active_orderedfilo==1],
                    D_orderedfilo[active_orderedfilo==1]*50+2,
                    'k')
        plt.scatter(location_in_link_orderedfilo[active_orderedfilo==0],
                    effectiveheight_orderedfilo[active_orderedfilo==0],
                    D_orderedfilo[active_orderedfilo==0]*50+2,
                    'grey')
        # Shade all pulse red/pink
        plt.scatter(location_in_link_orderedfilo[source_orderedfilo=='pulse'],
                    effectiveheight_orderedfilo[source_orderedfilo=='pulse'],
                    D_orderedfilo[source_orderedfilo=='pulse']*50+2,
                    'r',
                    alpha=0.3)
        
        text = plt.text(0.8,0.9,str(np.int64(tstep*(dt/(60*60*24))))+" Days") 
        
        plt.xlim(0,1)
        plt.ylim(0,4)
        plt.xlabel('Fractional distance down reach')
        plt.ylabel('Height above bedrock (m)')
        writer.grab_frame()
        plt.clf()
    plt.figure(figPulseAnim)
    plt.close()
    writer.finish()
#%%    Variable saving

source= parcels.dataset.source.values
active_layer = parcels.dataset.active_layer.values
location_in_link= parcels.dataset.location_in_link.values
D_= parcels.dataset.D.values
D_pulse= parcels.dataset.D.values[parcels.dataset['source'].values == 'pulse']
D_bed_sed= parcels.dataset.D.values[parcels.dataset['source'].values == 'initial_bed_sed']
volumes = parcels.dataset.volume.values
element_ids = parcels.dataset.element_id.values
recycle = parcels.dataset["recycle_destination"].values

element_left = parcels.dataset.element_id.values == -2
num_pulse_left = np.sum(element_left[pulse_mask, :])

mx_elev= np.max(Elev_change_DS)

runfiles_name = f"IOA_VARIABLES_1_{scenario}.npz"
np.savez("IOA_VARIABLES_1_SCEN1.npz", mask_here=mask_here, time_arrival=time_arrival, volumes=volumes,current_link=current_link, this_links_parcels=this_links_parcels,
          time_arrival_sort=time_arrival_sort,
          parcel_id_time_sorted=parcel_id_time_sorted, vol_ordered_filo=vol_ordered_filo, cumvol_orderedfilo=cumvol_orderedfilo,
          effectiveheight_orderedfilo=effectiveheight_orderedfilo, source_orderedfilo=source_orderedfilo, active_orderedfilo=active_orderedfilo,
        location_in_link_orderedfilo=location_in_link_orderedfilo, D_orderedfilo=D_orderedfilo)


runfiles_name = f"IOA_VARIABLES_2_{scenario}.npz"
np.savez(runfiles_name, sediment_active_percent=sediment_active_percent, num_pulse_each_link_DS=num_pulse_each_link_DS, 
         num_total_parcels_each_link_DS=num_total_parcels_each_link_DS, num_active_pulse_nozero_DS=num_active_pulse_nozero_DS, 
         vol_pulse_nozero_DS=vol_pulse_nozero_DS, vol_nozero_DS=vol_nozero_DS, D_mean_pulse_each_link_nozero_DS=D_mean_pulse_each_link_nozero_DS,
         percent_active_pulse_DS=percent_active_pulse_DS, percent_active_DS=percent_active_DS, Elev_change_DS=Elev_change_DS, Sed_flux=Sed_flux,
         days=days, **newer_pulse_D_by_timestep, D_pulse_initial = D_pulse_initial,D_pulse=D_pulse, D_bed_sed=D_bed_sed,
         dist_downstream_nodes_DS=dist_downstream_nodes_DS, dist_downstream_DS=dist_downstream_DS, D50_each_link=D50_each_link,
         sorted_initial_D=sorted_initial_D, cumulative_initial_pulse=cumulative_initial_pulse, sorted_final_D=sorted_final_D, cumulative_final_pulse=cumulative_final_pulse, 
         Slope_through_time_DS=Slope_through_time_DS, Change_D50s_each_link_DS=Change_D50s_each_link_DS, Change_Dmean_each_link_DS=Change_Dmean_each_link_DS, 
         percent_pulse_added=percent_pulse_added, n_recycled_parcels=n_recycled_parcels, 
         nonzero_travel_dist=nonzero_travel_dist, Sand_fraction_active=Sand_fraction_active, d_recycled_parcels=d_recycled_parcels,
         elev_change_at_link=elev_change_at_link, final_vol_on_link=final_vol_on_link, Final_vol_on_link_DS=Final_vol_on_link_DS,
         index_sorted_area_link=index_sorted_area_link, index_sorted_area_node=index_sorted_area_node,
         slope_DS=slope_DS, length=length, sed_act_vol=sed_act_vol,new_D = new_D, vol_pulse_abraded= vol_pulse_abraded,
         sed_total_vol=sed_total_vol, volume_pulse_at_each_link=volume_pulse_at_each_link, vol_pulse_exited = vol_pulse_exited, vol_pulse_on= vol_pulse_on,
         D_mean_pulse_each_link=D_mean_pulse_each_link, elev_change= elev_change, weighted_mean_velocity= weighted_mean_velocity, source= source, active_layer = active_layer,
         location_in_link = location_in_link, D_= D_, volumes = volumes, element_ids=element_ids, recycle= recycle, depth_DS=depth_DS,
         width_DS=width_DS, topo_DS=topo_DS, area_link_DS= area_link_DS, area_node_DS = area_node_DS, initial_vol_on_link= initial_vol_on_link
          ) 

# Saving a text file of model characteristics
timestep_in_days = dt/86400
# Converting the time in seconds to a timestamp
text_file = os.path.join(new_dir, 'Run ('+scenario_num+ ').txt')   
file = open('IOA _Run stats'+scenario_num+ ').txt', 'w')
model_characteristics = ["This code is running for scenario %s" %scenario,  "total_num_initial_pulse_parcels %s" %total_num_initial_pulse_parcels, "This is the tau_c_multiplier %s" %tau_c_multiplier, 
                          "This is the number of timesteps %s" %timesteps, "The number of days in each timestep is %s" % timestep_in_days, # length of dt in days
                          "The pulse is added at timestep %s" %pulse_time, "initial_num_pulse_parcels_by_vol %s" %initial_num_pulse_parcels_by_vol,
                          "The volume of each pulse parcel is %s" %pulse_parcel_vol, "This is the number of pulse parcels added to the network %s"
                          %total_num_added_pulse_parcels, "num_pulse_left= %s"  %num_pulse_left, "max ele %s" %mx_elev,
                          
                          "The current stage of this code is editing variables to be normalized since steady state."]
for line in model_characteristics:
    file.write(line)
    file.write('\n')
file.close()