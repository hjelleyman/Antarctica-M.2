import numpy as np
import os
import matplotlib.pyplot as plt
size = os.get_terminal_size()
import xarray as xr

def print_heading(heading):
    """Summary
    
    Parameters
    ----------
    heading : TYPE
        Description
    """
    sidespace = size[0]//4
    if len(heading)%2==1:
        heading = ' '+heading
    print('-'*((size[0]-2*sidespace)//2 * 2))
    print(' '*((size[0]-2*sidespace-2-len(heading))//2)+heading+' '*((size[0]-2*sidespace-2-len(heading))//2)+' ')
    print('-'*((size[0]-2*sidespace)//2 * 2))




def seaice_area_mean(seaice,n):
    area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
    seaice = seaice/250
    adjusted_sic = seaice*area

    return adjusted_sic.sum(dim = ('x','y'))
