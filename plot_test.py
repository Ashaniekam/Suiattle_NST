# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:11:57 2025

@author: longrea
"""

import matplotlib.pyplot as plt
import numpy as np

from landlab.grid.create_network import network_grid_from_raster, JustEndNodes
from landlab.components import (
    BedParcelInitializerUserD50,
    FlowDirectorSteepest,
    NetworkSedimentTransporter
)
from landlab.io import read_esri_ascii
from landlab.plot.graph import plot_links, plot_nodes
from landlab.plot.imshow import imshow_grid
from landlab.plot import plot_network_and_parcels

grid, z = read_esri_ascii("10m_hillshade_suia.asc", name="topographic__elevation")
grid.status_at_node[grid.nodes_at_right_edge] = grid.BC_NODE_IS_FIXED_VALUE
grid.status_at_node[np.isclose(z, -9999.0)] = grid.BC_NODE_IS_CLOSED

network_grid = network_grid_from_raster(
    grid,
    reducer=JustEndNodes(),
    minimum_channel_threshold=12000.0,
    include=["drainage_area", "topographic__elevation"],
)

network_grid.at_link["channel_width"]=np.ones(network_grid.number_of_links)
network_grid.at_link["reach_length"]=np.ones(network_grid.number_of_links)
network_grid.at_link["flow_depth"]=np.ones(network_grid.number_of_links)
network_grid.at_node["bedrock__elevation"]=network_grid.at_node["topographic__elevation"].copy()

initialize_parcels = BedParcelInitializerUserD50(
    network_grid,
    user_d50=0.05,
)

parcels = initialize_parcels()

fd = FlowDirectorSteepest(network_grid, "topographic__elevation")
fd.run_one_step()

nst = NetworkSedimentTransporter(network_grid,parcels,fd,)

nst.run_one_step(1)

plt.figure("DEM_and_Network")
imshow_grid(
    grid,
    "topographic__elevation",
    color_for_closed=None,
    allow_colorbar=False,
)

plot_network_and_parcels(
    network_grid,
    parcels,
    parcel_time_index=0,
    link_attribute=('sediment_total_volume'),
    parcel_alpha=0,
    network_linewidth= 3,
    link_attribute_title= "Volume",
    network_cmap= "coolwarm",
    fig=plt.figure("DEM_and_Network"),
    )