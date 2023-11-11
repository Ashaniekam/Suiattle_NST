# -*- coding: utf-8 -*-
"""
Spyder Editor

This is the code for my thesis on the Suiattle River

@author: longrea, pfeiffea

Goal: Sort out new Suiattle shapefile issues

"""


import matplotlib.pyplot as plt
import matplotlib.colors as colors
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

OUT_OF_NETWORK = NetworkModelGrid.BAD_INDEX - 1

# #### version #######
version = "new"

with_abrasion = 0

#read in shapefiles
shp_file = os.path.join(os.getcwd (), ("Suiattle_river_"+version+".shp"))
points_shapefile = os.path.join(os.getcwd(), ("Suiattle_nodes_"+version+".shp"))


#read in discharge from an excel file -  XXX_AP_9/28/23 - replacing this with grid.at_link["discharge"] - see Ln ~75


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

grid.at_link["discharge"]= grid.at_link["drainage_area"]*(230/np.max(grid.at_link["drainage_area"])) # 230 m3/s is the discharge value used in your calculations, I believe, though check me!

# Access the coordinates: finding the x coordinates of the nodes in the network
x_of_node = grid.node_x

#Using drainage area for links and nodes to create ordering indices
area_link = grid.at_link["drainage_area"]
area_node = grid.at_node["drainage_area"]

index_sorted_area_link = (area_link).argsort()
index_sorted_area_node = (area_node).argsort()

#Sorted Drainage Area
area_link = area_link[index_sorted_area_link]
area_node = area_node[index_sorted_area_node]

#Sorted Topographic Elevation and Bedrock Elevation
topo = grid.at_node['topographic__elevation']
topo = topo[index_sorted_area_node]
grid.at_node["bedrock__elevation"] = topo.copy()

#Sorted Width
width = grid.at_link["channel_width"]
width = width[index_sorted_area_link]

#Sorted Slope
slope = grid.at_link["channel_slope"]
slope = slope[index_sorted_area_link]

#Sorted Length
length = grid.at_link["reach_length"]
length = length[index_sorted_area_link]
cum_sum_upstream = np.cumsum(length)

grid.at_link["dist_upstream"]= cum_sum_upstream
dist_upstream = grid.at_link["dist_upstream"]/1000
dist_upstream_nodes = np.insert(dist_upstream, 0, 0.0)

plt.figure()
plt.plot(dist_upstream, grid.at_link["channel_slope"], '-' )
plt.title("Starting channel Slope")
plt.ylabel("Slope")
plt.xlabel("Distance downstream (km)")
plt.show()


#start for scaling sediment feed by drainage area
#calculates the difference between the upstream and downstream dranage area for each link
diff_area_at_each_link =  []
for d in range ((len(area_link)-1)):
    diff_area = area_link[d] - area_link[d+1]
    diff_area_at_each_link.append(diff_area)
#to calculate flow depth

Mannings_n = 0.086 #0.1734 #median calculated from Suiattle gages (units m^3/s)
grid.at_link["flow_depth"] = ((Mannings_n*grid.at_link["discharge"])/ ((slope**0.5)*width))**0.6

depth = grid.at_link["flow_depth"].copy()

rho_water = 1000
rho_sed = 2650
gravity = 9.81
tau = rho_water * gravity * depth * slope


vol_of_channel = width*length*depth

number_of_links = grid.number_of_links


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
plt.title("SECOND channel Slope")
plt.ylabel("Slope")
plt.xlabel("Distance downstream (km)")
plt.show()

#creating sediment parcels in the DATARECORD

slope_depend_Shields = 0.15* slope**0.25
tau_c_multiplier = 2.5#to change grain size
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


# XXX TIME in the model

timesteps = 5 # number of timesteps

#dt = 60*60*24*1  # len of timesteps (1 day)
dt = 60*60*24*7  # len of timesteps (seconds)
len_dt = dt/86400

n_lines = 10 # note: # timesteps must be divisible by n_lines with timesteps%n_lines == 0
color = iter(plt.cm.viridis(np.linspace(0, 1, n_lines+1)))

# flow direction
fd = FlowDirectorSteepest(grid, "topographic__elevation")
fd.run_one_step()

# initialize the networksedimentTransporter

nst = NetworkSedimentTransporter(
    grid,
    parcels,
    fd,
    bed_porosity=0.3,
    g=9.81,
    fluid_density=1000,
    transport_method="WilcockCrowe",
    active_layer_method = "Constant10cm",
    active_layer_d_multiplier= 0.5,
)

# run model in time
pulse_time = 2
pulse_parcel_vol = 5 
num_pulse_parcels = 1000 #np.round((np.mean(vol_of_channel/2))/pulse_parcel_vol) #980000 

effective_pulse_depth = (num_pulse_parcels*pulse_parcel_vol)/(grid.at_link["reach_length"][1]*grid.at_link["channel_width"][1]) #volume/(length*width)

Pulse_cum_transport = []

thickness = ((np.nanmedian(parcels.dataset["volume"].values))*median_number_of_starting_parcels)/(grid.at_link["reach_length"]*width)

#variables needed for plots
# XXX
n_recycled_parcels = np.empty(timesteps)
d_recycled_parcels = np.empty(timesteps)
time_elapsed = np.empty(timesteps)
count_recyc = np.empty(timesteps)
timestep_array = np.arange(timesteps)
sed_total_vol = np.empty([grid.number_of_links,timesteps])
sediment_active_percent = np.empty([grid.number_of_links,timesteps])
active_d_mean= np.empty([grid.number_of_links,timesteps])
num_active_pulse= np.empty([grid.number_of_links,timesteps])
volume_pulse_at_each_link = np.empty([grid.number_of_links,timesteps])
D_mean_pulse_each_link= np.empty([grid.number_of_links,timesteps])
num_pulse_each_link= np.empty([grid.number_of_links,timesteps], dtype=int)
num_total_parcels_each_link= np.empty([grid.number_of_links,timesteps], dtype=int)
num_count_recyc_parc= np.empty([grid.number_of_links,timesteps], dtype=int)
num_recyc_each_link= np.empty([grid.number_of_links,timesteps], dtype=int)
time_array= np.arange(1, timesteps + 1)

Pulse_cum_transport = np.ones([num_pulse_parcels,1])*np.nan
Elev_change = np.empty([grid.number_of_nodes,timesteps]) # use cap first letter to denote 2d 
num_parcels = parcels.number_of_items #total number of parcels in network
count_recyc_parc = []
      
for t in range(0, (timesteps*dt), dt):
    #by index of original sediment
    start_time = model_time.time()

    total_number_of_parcels = parcels.dataset.element_id.values[:, -1].astype(int)
    # XXX
    
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
    
    #XXX
    #figure out how to plot the number of parcels recycled to each link
    
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
        
        pulse_location = [random.randrange(17, 22, 1) for i in range(num_pulse_parcels)] #code from nst_test.py: adding pulse randomly near the fan
        newpar_element_id = pulse_location
        newpar_element_id = np.expand_dims(newpar_element_id, axis=1)
        
        
        new_starting_link = np.squeeze(newpar_element_id)
        
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
        
        new_density = 2650 * np.ones(np.size(newpar_element_id))  # (kg/m3)
        
        new_location_in_link = np.random.rand(
            np.size(newpar_element_id), 1
        )
        if with_abrasion == 1: 
            new_abrasion_rate = 0.00004 * np.ones(np.size(newpar_element_id)) 
        else: 
            new_abrasion_rate = 0 * np.ones(np.size(newpar_element_id))
            
        new_D = np.random.lognormal(np.log(0.03),np.log(1.5),np.shape(newpar_element_id))  
        # (m) the diameter of grains in each parcel
        #original::  new_D = np.random.lognormal(np.log(0.03),np.log(3),np.shape(newpar_element_id))

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
                        nst._distance_traveled_cumulative[-num_pulse_parcels:],
                        axis = 1),
                axis = 1
                )
    if t == parcels.dataset.time[0].values: # if we're at the first timestep

        avg_init_sed_thickness = np.mean(
            grid.at_node['topographic__elevation'][:-1]-grid.at_node['bedrock__elevation'][:-1])        
        grid.at_node['bedrock__elevation'][-1]=grid.at_node['bedrock__elevation'][-1]+avg_init_sed_thickness


        elev_initial = grid.at_node['topographic__elevation'].copy()
        
    if t%((timesteps*dt)/n_lines)==0:        
    
        # # PLOT 1: elevation change through time, on long profile
        plt.figure("Tracking elev change")
        
        elev_change = grid.at_node['topographic__elevation']-elev_initial
        
     
        c = next(color)
        plt.plot(dist_upstream_nodes, elev_change, c=c)
        plt.xlabel('distance downstream (km)')
        plt.ylabel('elevation change (m)')
        

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
   
        
    element_pulse = parcels.dataset.element_id.values[:, -1].astype(int)
    element_pulse= element_pulse[mask_ispulse]
    aggregate_parcels_at_link_sum(
        num_active_pulse[:,np.int64(t/dt)],
        number_of_links, 
        element_pulse,
        len(parcels.dataset.active_layer.values[mask_ispulse, -1]),
        parcels.dataset.active_layer.values[mask_ispulse, -1],
    )

    
    sorted_num_active_pulse = num_active_pulse[index_sorted_area_link]
    ## Plot 7: D_MEAN_ACTIVE through time
    active_d_mean [:,np.int64(t/dt)]= nst.d_mean_active
    
   
    #plot8: Volume of pulse parcels through time
    
    aggregate_parcels_at_link_sum(
        volume_pulse_at_each_link[:,np.int64(t/dt)],
        number_of_links, 
        element_pulse,
        len(parcels.dataset.volume.values[mask_ispulse, -1]),
        parcels.dataset.volume.values[mask_ispulse, -1],
    )
    sorted_volume_pulse_at_each_link = volume_pulse_at_each_link[index_sorted_area_link]
    
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
    
    #PLOT 10: number of pulse parcel each link
    aggregate_parcels_at_link_count(
        num_pulse_each_link[:,np.int64(t/dt)],
        num_pulse_each_link.shape[0],
        element_pulse,
        len(element_pulse),
    )
    sorted_num_pulse_each_link =  num_pulse_each_link[index_sorted_area_link]
    # Plot #: number of total parcels each link
    aggregate_parcels_at_link_count(
         num_total_parcels_each_link[:,np.int64(t/dt)],
         num_total_parcels_each_link.shape[0],
         total_number_of_parcels,
         len(total_number_of_parcels),
         
     )
    sorted_num_total_parcels_each_link = num_total_parcels_each_link[index_sorted_area_link]
    #XXX 

parcels_dataframe= parcels.dataset.to_dataframe()
parcels_dataframe.to_csv('Parcels_data.csv', sep=',', index=True) 

plt.figure()
plt.plot(area_link, num_recycle_bed, c=c)
plt.title("count_recyc_parc (Timesteps = %i)" %timesteps)
plt.xlabel("Drainage area")
plt.ylabel("Number of recycled parcels")
plt.savefig('count_recyc_parcpng')

#XXX

# plt.figure()
# plt.plot(count_recyc)
# plt.ylabel('Number of recycled parcels')
# plt.xlabel('Drainage Area')
# plt.show()

#print("This runs to 1000 timesteps to check if A changes from not being empty")

#Saving a text file of model characteristics
# Converting the time in seconds to a timestamp
c_ti = model_time.ctime(os.path.getctime("nst_SuiattleRiver.py"))
file = open('Suiattle_run_characteristics.txt', 'w')
model_characteristics = ["This script was created at %s" %c_ti, "This is the tau_c_multiplier %s" %tau_c_multiplier, 
                          "This is the number of timesteps %s" %timesteps, "The number of days in each timestep is %s" %len_dt,
                          "The pulse is added at timestep %s" %pulse_time,
                          "The volume of each pulse parcel is %s" %pulse_parcel_vol, "This is the number of pulse parcels added to the network %s"
                          %num_pulse_parcels,
                          "This is the abarsion rate of the pulse %s" % new_abrasion_rate[0],
                          "The current stage of this code is using Allison message (email) to test if the sediment is being added based on drainage area"]
for line in model_characteristics:
# file.write('%d' % link_len)
    file.write(line)
    file.write('\n')
file.close()
# %% ####   Plotting and analysis of results    ############  

# Print some pertinent information about the model run
print('mean grain size of pulse',np.mean(
    parcels.dataset.D.values[parcels.dataset.source.values[:] == 'pulse',-1]))


# final volume of sediment on each link

final_vol_on_link = np.empty(number_of_links, dtype = float)
aggregate_parcels_at_link_sum(
    final_vol_on_link,
    number_of_links,
    parcels.dataset.element_id.values[:, -1].astype(int),
    len(parcels.dataset.volume.values[:, -1]),
    parcels.dataset.volume.values[:, -1],
)
sorted_final_vol_on_link = final_vol_on_link[index_sorted_area_link] 
######### Plots ###########

#Plot 1 v2
elev_change_at_link = ((final_vol_on_link - initial_vol_on_link) /
                        (grid.at_link["reach_length"]*grid.at_link["channel_width"]))  # volume change in link/ link bed area

links=np.arange(grid.number_of_links)
plt.figure()
plt.plot(elev_change_at_link)
plt.title("Change in elevation through time (Timesteps = %i)" %timesteps)
plt.ylabel("Elevation change (m)")
plt.savefig('TrackElevChangev2 (1% abrasion).png')

#Plot 2
# #### Tracking the output: are we at (or near) steady state?  ####
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(timestep_array, d_recycled_parcels,'-', color='brown')
ax2.plot(timestep_array, n_recycled_parcels,'-', color='k') #,'.' makes it a dot chart, '-' makes line chart

ax1.set_xlabel("Timesteps")
ax1.set_ylabel("Mean D recycled parcels (m)", color='k')
plt.title("Recycled parcels (Timesteps = %i)" %timesteps)
ax2.set_ylabel("Number of recycled parcels", color='brown')
plt.savefig('Recycled (1% abrasion).png')
plt.show()

link_len = np.mean(grid.length_of_link)

#Plot 3
# ####  How far is the pulse material transporting?    ####

d_percent_loss = (parcels.dataset.D.values[mask_ispulse,-1] - new_D[:,0])/new_D[:,0]
plt.scatter(parcels.dataset["D"].values[-num_pulse_parcels:,-1],Pulse_cum_transport[:,-1],10,d_percent_loss, vmin=-0.1, vmax=-0.6)
plt.title("How far is the pulse material transporting? (Timesteps = %i)" %timesteps)
plt.ylabel('Pulse parcel cumulative transport distance (m)')
plt.xlabel('D (m)')
plt.colorbar(label='Percentage of grains loss (%)')
plt.ylim(0, 100000)
# reference lines
plt.axhline(y = link_len, color = 'grey', linestyle = ':') # plot a horizontal line
plt.text(0.06,link_len*1.1,'one link length')

plt.axhline(y = link_len*grid.number_of_links, color = 'grey', linestyle = ':') # plot a horizontal line
plt.text(0.06,link_len*grid.number_of_links*0.95,'full channel length')
plt.savefig('DistTravelPulse(1% abrasion).png')
plt.show()

#Plot 4
# How far have the non-pulse parcels traveled in total?
travel_dist = nst._distance_traveled_cumulative[:-num_pulse_parcels]
nonzero_travel_dist = travel_dist[travel_dist>0]
plt.hist(np.log10(nonzero_travel_dist),30) # better to make log-spaced bins...
plt.xlabel('log10( Cum. parcel travel distance (m) )')
plt.ylabel('Number of non-pulse parcels')
plt.title("How far have the non-pulse parcels traveled in total? (Timesteps = %i)" %timesteps)
plt.savefig('DistTravelNonPulse(1% abrasion).png')
plt.text(2,50, 'n = '+str(len(nonzero_travel_dist))+' parcels')
# %%

#Plot 4
# What is the active parcel volume/total?
plt.figure()

#plt.plot(grid.at_link["dist_upstream"],sediment_active_percent)
plt.pcolor(time_array, dist_upstream, sed_total_vol)
plt.title("Active parcel volume (Timesteps = %i)" %timesteps)
plt.xlabel('Timesteps')
plt.ylabel('Distance from upstream (km)')
plt.savefig('percentActive.png')
plt.show()

plt.figure()
plt.pcolor(time_array, dist_upstream,sediment_active_percent)
plt.colorbar(label='average % active parcels')
plt.title("Active parcel volume (Timesteps = %i)" %timesteps)
plt.xlabel('Timesteps')
plt.ylabel('Distance from upstream (km)')
plt.savefig('percentActive.png')
plt.show()

#Plot 5
#what is the number of parcels active per link
plt.figure()
plt.plot(sorted_num_active_parcels_each_link)
plt.title("Number of parcels active (Timesteps = %i)" %timesteps)
plt.xlabel("Link number downstream")
plt.ylabel("Number of active parcels")
plt.savefig('ActiveParcel.png')
plt.show()

# XXX
plt.figure()
plt.pcolor(time_array, dist_upstream, sorted_num_pulse_each_link)
plt.title("Number of total pulse parcels (Timesteps = %i)" %timesteps)
plt.colorbar(label='Num pulse parcel')
plt.xlabel("Timesteps")
plt.ylabel("Distance from upstream (km)")
plt.savefig('Total_pulseParcel.png')
plt.show()

plt.figure()
plt.pcolor(time_array, dist_upstream, sorted_num_total_parcels_each_link)
plt.title("Number of total parcels (Timesteps = %i)" %timesteps)
plt.colorbar(label='Number of parcels')
plt.xlabel("Timesteps")
plt.ylabel("Distance from upstream (km)")
plt.savefig('TotalParcel.png')
plt.show()


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(dist_upstream, D50_each_link,'-', color='k')
ax2.plot(dist_upstream, sorted_num_total_parcels_each_link,'-', color='brown') #,'.' makes it a dot chart, '-' makes line chart

ax1.set_xlabel("Distance from upstream (km)")
ax1.set_ylabel("D50", color='k')
plt.title("Number of parcels and the grain size (Timesteps = %i)" %timesteps)
ax2.set_ylabel("Number of total parcels", color='brown')
plt.savefig('RSm (1% abrasion).png')
plt.show()

#%%
#Plot 6
#what is the  number of pulse parcels in the active layer

plt.figure()
num_active_pulse_nozero = sorted_num_active_pulse.copy()
num_active_pulse_nozero[num_active_pulse_nozero == 0]= np.nan


plt.pcolor(time_array, dist_upstream, num_active_pulse_nozero, cmap= 'winter_r')
plt.title("How far is the pulse material transporting? (Timesteps = %i)" %timesteps)
plt.colorbar(label='Number of active pulse parcels')
plt.xlabel("Timesteps")
plt.ylabel("Distance from upstream (km)")
plt.savefig('ActivePluse(1% abrasion).png', dpi=700)
plt.show()
#%%

#Plot 7
#what is d_mean_active downstream
plt.figure()
plt.pcolor(time_array, dist_upstream, active_d_mean)
plt.title("What is the mean grain size through time? (Timesteps = %i)" %timesteps)

plt.colorbar(label='Mean grain size (m)')
plt.xlabel("Timesteps")
plt.ylabel("Link downstream")
plt.savefig('ActiveDmean(1% abrasion).png')
plt.show()

#%% #plot8: Volume of pulse parcels through time 
plt.figure()
vol_pulse_nozero = sorted_volume_pulse_at_each_link.copy()
vol_pulse_nozero[vol_pulse_nozero == 0]= np.nan
plt.pcolor(time_array, dist_upstream,vol_pulse_nozero, cmap= 'plasma_r', vmin=0, vmax=5000)
plt.colorbar(label='Volume of pulse parcels ($m^3$)')
plt.title("Volume of pulse parcel through time (Timesteps = %i)" %timesteps)
plt.xlabel("Timesteps")
plt.ylabel("Distance from upstream (km)")
plt.savefig('Vol_pluse(1% abrasion).png', dpi=700)
plt.show()

#PLOT 9: D mean of pulse at each link
plt.figure()
D_mean_pulse_each_link_nozero= sorted_D_mean_pulse_each_link.copy()
D_mean_pulse_each_link_nozero[D_mean_pulse_each_link_nozero == 0]= np.nan
plt.pcolor(time_array, dist_upstream, D_mean_pulse_each_link_nozero, cmap= 'winter_r', vmin=0.0, vmax=0.10)
#plt.title("Dmean of pulse parcel through time (1% abrasion)")
#norm=colors.LogNorm(vmin=.01, vmax=.1),
plt.colorbar(label='Mean grain size of pulse parcels (m)')
plt.title("Dmean grain size through time (Timesteps = %i)" %timesteps)
plt.xlabel("Timesteps")
plt.ylabel("Distance from upstream (km)")
plt.savefig('Dmean_pluse(1% abrasion).png', dpi=700)
plt.show()

#PLOT 10: percentage active parcels
plt.figure()
percent_active= num_active_pulse/ num_pulse_each_link
sorted_percent_active = percent_active[index_sorted_area_link] 
plt.pcolor(time_array, dist_upstream, sorted_percent_active, cmap= 'Wistia', vmin=0.0, vmax=1.0)
plt.colorbar(label='Percentage of active pulse parcels (%)')
plt.title("Percentage active pulse parcels (Timesteps = %i)" %timesteps)
plt.xlabel("Timesteps")
plt.ylabel("Distance from upstream (km)")
plt.savefig('percent_active_pulse(1% abrasion).png', dpi=700)
plt.show()
#%%

# PLOT 1 v3: elevation change through time, as pcolor
plt.figure(" Track elev change")
plt.pcolor(time_array, dist_upstream_nodes, Elev_change, shading='auto', norm=colors.CenteredNorm(), cmap='coolwarm') 
plt.colorbar(label='Elevation change from initial (m)')
plt.title("Elevation change through time (Timesteps = %i)" %timesteps)
plt.xlabel('Timesteps')
plt.ylabel('Distance from upstream (km)')
plt.savefig('TrackElevChange(1% abrasion).png', dpi=700)
plt.show()
