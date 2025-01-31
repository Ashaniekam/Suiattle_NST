# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:24:56 2025

@author: longrea
"""
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.animation as animation
# %% Pulse percent bars
link = 7
output_folder = os.path.join(os.getcwd (), ("Gifs replace this folder"))
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
# Initiate an animation writer using the matplotlib module, `animation`.
figPulseAnim, axPulseAnim = plt.subplots(1, 1, dpi=600)
writer = animation.FFMpegWriter(fps=50)
gif_name = output_folder +"/"+ scenario_num + "--Parcel_evol_at_link" + str(link) +".gif"
writer.setup(figPulseAnim,gif_name)
for tstep in range(timesteps):
# for tstep in [500]: # for now, doing for just one timestep
    mask_here = parcels.dataset.element_id.values[:,tstep] == link
    time_arrival = parcels.dataset.time_arrival_in_link.values[mask_here, tstep]
    volumes = parcels.dataset.volume.values[:, tstep]
    current_link = parcels.dataset.element_id.values[:,tstep].astype(int)
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
                D_orderedfilo[active_orderedfilo==1]*50+5,
                'k')
    plt.scatter(location_in_link_orderedfilo[active_orderedfilo==0],
                effectiveheight_orderedfilo[active_orderedfilo==0],
                D_orderedfilo[active_orderedfilo==0]*50+5,
                'grey')
    # Shade all pulse red/pink
    plt.scatter(location_in_link_orderedfilo[source_orderedfilo=='pulse'],
                effectiveheight_orderedfilo[source_orderedfilo=='pulse'],
                D_orderedfilo[source_orderedfilo=='pulse']*50+5,
                'r',
                alpha=0.7)
    text = plt.text(0.8,0.9,str(np.int64(tstep*dt/(60*60*24)))+" Days") #sloppy workaround, but I'm on the plane
    plt.xlim(0,1)
    plt.ylim(0,4)
    plt.xlabel('Fractional distance down reach')
    plt.ylabel('Height above bedrock (m)')
    writer.grab_frame()
    plt.clf()
plt.figure(figPulseAnim)
plt.close()
writer.finish()






