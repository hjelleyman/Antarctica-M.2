import numpy as np
import os
import matplotlib.pyplot as plt
import xarray as xr

from plyer import notification

def print_heading(heading):
    """Summary
    
    Parameters
    ----------
    heading : TYPE
        Description
    """
    
    size = os.get_terminal_size()
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


def notify(message):
    notification.notify(
        title='Penguin Code',
        message=message,
        app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
        timeout=10,  # seconds
        )

def savefigures(folder=None,filename=None):
    if not os.path.isdir(f'{folder}/hres'):
        os.makedirs(f'{folder}/hres')
    if not os.path.isdir(f'{folder}/lres'):
        os.makedirs(f'{folder}/lres')
    plt.savefig(f'{folder}/hres/{filename}.pdf', bbox_inches='tight')
    plt.savefig(f'{folder}/lres/{filename}.png', bbox_inches='tight')