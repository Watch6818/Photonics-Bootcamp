'''
Scripted layout for ring resonators using SiEPIC-Tools
in the SiEPIC-EBeam-PDK "EBeam" technology

by Lukas Chrostowski, 2025 

- Silicon waveguides
- Several ring resonators with different radii and gap
  
Use instructions:

Run in Python, e.g., VSCode

pip install required packages:
 - klayout, SiEPIC, siepic_ebeam_pdk, numpy

'''

designer_name = 'GarrettDavis'
top_cell_name = 'EBeam_%s_rings' % designer_name
export_type = 'static'  # static: for fabrication, PCell: include PCells in file

import pya
from pya import *

import SiEPIC
from SiEPIC._globals import Python_Env
from SiEPIC.scripts import zoom_out, export_layout
from SiEPIC.verification import layout_check
import os
import numpy

if Python_Env == 'Script':
    try:
        # For external Python mode, when installed using pip install siepic_ebeam_pdk
        import siepic_ebeam_pdk
    except:
        # Load the PDK from a folder, e.g, GitHub, when running externally from the KLayout Application
        import os, sys
        path_GitHub = os.path.expanduser('~/Documents/GitHub/')
        sys.path.insert(0,os.path.join(path_GitHub, 'SiEPIC_EBeam_PDK/klayout'))
        import siepic_ebeam_pdk

tech_name = 'EBeam'

# Layout function
def dbl_bus_ring_res():

    # Import functions from SiEPIC-Tools
    from SiEPIC.extend import to_itype
    from SiEPIC.scripts import connect_cell, connect_pins_with_waveguide
    from SiEPIC.utils.layout import new_layout, floorplan

    # Create a layout for testing a double-bus ring resonator.
    # uses:
    #  - the SiEPIC EBeam Library
    # creates the layout in the presently selected cell
    # deletes everything first
    
    # Configure parameter sweep  
    pol = 'TE'
    sweep_radius = [13, 15,]
    sweep_gap    = [0.1, 0.1]
    x_offset = 67
    wg_bend_radius = 20

    wg_width = 0.5
    max_radius = max(sweep_radius)
    '''
    Create a new layout using the EBeam technology,
    with a top cell
    and Draw the floor plan
    '''    
    cell, ly = new_layout(tech_name, top_cell_name, GUI=True, overwrite = True)
    floorplan(cell, 605e3, 410e3)

    if SiEPIC.__version__ < '0.5.1':
        raise Exception("Errors", "This example requires SiEPIC-Tools version 0.5.1 or greater.")

    # Layer mapping:
    LayerSiN = ly.layer(ly.TECHNOLOGY['Si'])
    fpLayerN = cell.layout().layer(ly.TECHNOLOGY['FloorPlan'])
    TextLayerN = cell.layout().layer(ly.TECHNOLOGY['Text'])
    
    
    # Create a sub-cell for our Ring resonator layout
    top_cell = cell
    dbu = ly.dbu
    cell = cell.layout().create_cell("RingResonator")
    t = Trans(Trans.R0, 40 / dbu, 14 / dbu)

    # place the cell in the top cell
    top_cell.insert(CellInstArray(cell.cell_index(), t))
    
    # Import cell from the SiEPIC EBeam Library
    cell_ebeam_gc = ly.create_cell("GC_%s_1550_8degOxide_BB" % pol, "EBeam")
    # get the length of the grating coupler from the cell
    gc_length = cell_ebeam_gc.bbox().width()*dbu
    # spacing of the fibre array to be used for testing
    GC_pitch = 127
    instGCs = []
    # Build GCs to be in a single column on the left
    for j in range(4):
            x = 0
            t = Trans(Trans.R0, to_itype(x,dbu), j*127/dbu)
            instGCs.append(cell.insert(CellInstArray(cell_ebeam_gc.cell_index(), t)) )

    # Opt-in label at 4th grating coupler
    t = Trans(Trans.R90, to_itype(x,dbu), to_itype(GC_pitch*2,dbu))
    text = Text("opt_in_%s_1550_device_%s_FilterBank_R13_14_15_g100" % (pol.upper(), designer_name), t)
    text.halign = 1
    cell.shapes(TextLayerN).insert(text).text_size = 5/dbu

    prev_dc2 = None
    # Loop through the parameter sweep
    for i in range(len(sweep_gap)):
        # place layout at location:
        if i==0:
            x=0
        else:
            # Next device is placed at the right-most element + length of the grating coupler
            # or 60 microns from the previous grating coupler, whichever is greater.
            # Get x position from first non-None GC, or use inst_dc2 bbox
            gc_x = next((gc.trans.disp.x * dbu for gc in instGCs if gc is not None), 0)
            x = inst_dc2.bbox().right * dbu + gc_length + 1
        
        # get the parameters
        r = sweep_radius[i]
        g = sweep_gap[i]
        
        # Label for automated measurements, laser on Port 2, detectors on Ports 1, 3, 4
                  
        # Ring resonator from directional coupler PCells
        cell_dc = ly.create_cell("ebeam_dc_halfring_straight", "EBeam", { "r": r, "w": wg_width, "g": g, "bustype": 0 } )
        y_ring = 2*GC_pitch - (cell_ebeam_gc.bbox().height()*dbu) - 2*(r - sweep_radius[0]) # not sure why this works
            # first directional coupler
        t1 = Trans(Trans.R0, to_itype(x+wg_bend_radius, dbu), to_itype(y_ring, dbu))
        inst_dc1 = cell.insert(CellInstArray(cell_dc.cell_index(), t1))
            # add 2nd directional coupler, snapped to the first one
        inst_dc2 = connect_cell(inst_dc1, 'pin2', cell_dc, 'pin4')
        
        # Create paths for waveguides, with the type defined in WAVEGUIDES.xml in the PDK
        waveguide_type='Strip TE 1550 nm, w=500 nm'
        
        # Top-left ring 1 and in-between rings connections
        if prev_dc2 is None:
            connect_pins_with_waveguide(instGCs[2], 'opt1', inst_dc2, 'pin3', waveguide_type=waveguide_type)
        else:
            connect_pins_with_waveguide(prev_dc2, 'pin1', inst_dc2, 'pin3', waveguide_type=waveguide_type)
        prev_dc2 = inst_dc2 # Store for next iteration

        # Top-right ring 3 connection
        if i == len(sweep_gap) - 1:
            connect_pins_with_waveguide(inst_dc2, 'pin1', instGCs[3], 'opt1', waveguide_type=waveguide_type)

        # Drop port (all rings)
        # Drop port - only rings 1 and 3
        if i == 0:  # Ring 1
            connect_pins_with_waveguide(instGCs[1], 'opt1', inst_dc1, 'pin1', waveguide_type=waveguide_type)
        else:  # Ring 2
            connect_pins_with_waveguide(instGCs[0], 'opt1', inst_dc1, 'pin1', waveguide_type=waveguide_type)

        # Add port terminator (all rings)
        cell_term = ly.create_cell("ebeam_terminator_te1550", "EBeam")
        # Place terminator below the current ring at y=0, offset slightly from ring center
        # t_term = Trans(Trans.R0, to_itype(x + wg_bend_radius, dbu), 0)
        # inst_term = cell.insert(CellInstArray(cell_term.cell_index(), t_term))
        inst_term = connect_cell(inst_dc1, 'pin3', cell_term, 'opt1')
        # connect_pins_with_waveguide(inst_dc1, 'pin3', inst_term, 'opt1', waveguide_type=waveguide_type)

    return ly, cell
    
ly, cell = dbl_bus_ring_res()

# Verify
num_errors = layout_check(cell=cell, verbose=False, GUI=True)
print('Number of errors: %s' % num_errors)

# Export for fabrication, removing PCells
path = os.path.dirname(os.path.realpath(__file__))
filename, extension = os.path.splitext(os.path.basename(__file__))
if export_type == 'static':
    file_out = export_layout(cell, path, filename, relative_path = '..', format='oas', screenshot=True)
else:
    file_out = os.path.join(path,'..',filename+'.oas')
    ly.write(file_out)

from SiEPIC.verification import layout_check
print('SiEPIC_EBeam_PDK: example_Ring_resonator_sweep.py - verification')
file_lyrdb = os.path.join(path,filename+'.lyrdb')
num_errors = layout_check(cell = cell, verbose=False, GUI=True, file_rdb=file_lyrdb)

# Display the layout in KLayout, using KLayout Package "klive", which needs to be installed in the KLayout Application
if Python_Env == 'Script':
    from SiEPIC.utils import klive
    klive.show(file_out, lyrdb_filename=file_lyrdb, technology=tech_name)

print('layout script done')