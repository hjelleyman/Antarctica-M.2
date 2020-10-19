import xarray as xr
import numpy as np 
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import itertools
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import cartopy.crs as ccrs
from matplotlib import cm
from numba import cuda, float32
import time
import modules.week2 as w2

def load_indicies(indicies,temporal_resolution):
    """Loads in the different indicies we are interested in.
        
        1) IPO          - ipo.nc
        2) nina 3.4     - nina34.data
        3) nina 1+2     - nina12.data
        4) DMI          - dmi.nc
        5) SAM          - sam.txt
        6) MEI          - meiv2.data
        7) SOI          - soi.txt

    Output is a xarray DataSet which contains all the indicies listed above.
    """

    ds = {}


    # DMI
    if 'DMI' in indicies:
        ds['DMI'] = xr.open_dataset('Data/Indicies/dmi.nc').DMI
    

    # SAM
    if 'SAM' in indicies:
        sam = np.genfromtxt('Data/Indicies/newsam.1957.2007.txt', skip_header =1, skip_footer = 1)[:,1:]
        index = range(1957, 2020)
        columns = range(1,13)
        sam = pd.DataFrame(data = sam, columns = columns, index = index)
        sam = sam.stack().reset_index()
        sam.columns = ['year', 'month', 'SAM']
        sam['time'] = pd.to_datetime(sam.year*100+sam.month,format='%Y%m')
        sam = sam.set_index('time').SAM
        sam = xr.DataArray(sam)
        ds['SAM'] = sam

    # IPO
    if 'IPO' in indicies:
        ipo = np.genfromtxt('Data/Indicies/tpi.timeseries.ersstv5.data', skip_header = 1, skip_footer = 11)[:,1:]
        index = range(1854, 2021)
        columns = range(1,13)
        ipo = pd.DataFrame(data = ipo, columns = columns, index = index)
        ipo = ipo.stack().reset_index()
        ipo.columns = ['year', 'month', 'IPO']
        ipo['time'] = pd.to_datetime(ipo.year*100+ipo.month,format='%Y%m')
        ipo = ipo.set_index('time').IPO
        ipo = ipo[ipo>-10]
        ipo = xr.DataArray(ipo)
        ds['IPO'] = ipo

    if '-IPO' in indicies:
        ipo = np.genfromtxt('Data/Indicies/tpi.timeseries.ersstv5.data', skip_header = 1, skip_footer = 11)[:,1:]
        index = range(1854, 2021)
        columns = range(1,13)
        ipo = pd.DataFrame(data = ipo, columns = columns, index = index)
        ipo = ipo.stack().reset_index()
        ipo.columns = ['year', 'month', 'IPO']
        ipo['time'] = pd.to_datetime(ipo.year*100+ipo.month,format='%Y%m')
        ipo = ipo.set_index('time').IPO
        ipo = ipo[ipo>-10]
        ipo = xr.DataArray(ipo)
        ds['-IPO'] = -ipo
    # SOI
    if 'SOI' in indicies:
        SOI = np.genfromtxt('Data/Indicies/soi.txt', usecols=np.arange(1,13))
        index = range(1951 , 2021)
        columns = range(1,13)
        SOI = pd.DataFrame(data = SOI, columns = columns, index = index)
        SOI = SOI.stack().reset_index()
        SOI.columns = ['year', 'month', 'SOI']
        SOI['time'] = pd.to_datetime(SOI.year*100+SOI.month,format='%Y%m')
        SOI = SOI.set_index('time').SOI
        SOI = SOI[SOI>-10]
        SOI = xr.DataArray(SOI)
        ds['SOI'] = SOI

    # nina12
    if 'nina12' in indicies:
        nina12 = np.genfromtxt('Data/Indicies/nina12.data', usecols=np.arange(1,13))
        index = range(1950 , 2021)
        columns = range(1,13)
        nina12 = pd.DataFrame(data = nina12, columns = columns, index = index)
        nina12 = nina12.stack().reset_index()
        nina12.columns = ['year', 'month', 'nina12']
        nina12['time'] = pd.to_datetime(nina12.year*100+nina12.month,format='%Y%m')
        nina12 = nina12.set_index('time').nina12
        nina12 = nina12[nina12>-80]
        nina12 = xr.DataArray(nina12)
        ds['nina12'] = nina12

    # nina34
    if 'nina34' in indicies:
        nina34 = np.genfromtxt('Data/Indicies/nina34.data', usecols=np.arange(1,13))
        index = range(1950 , 2021)
        columns = range(1,13)
        nina34 = pd.DataFrame(data = nina34, columns = columns, index = index)
        nina34 = nina34.stack().reset_index()
        nina34.columns = ['year', 'month', 'nina34']
        nina34['time'] = pd.to_datetime(nina34.year*100+nina34.month,format='%Y%m')
        nina34 = nina34.set_index('time').nina34
        nina34 = nina34[nina34>-80]
        nina34 = xr.DataArray(nina34)
        ds['nina34'] = nina34

    # meiv2
    if 'meiv2' in indicies:
        meiv2 = np.genfromtxt('Data/Indicies/meiv2.data', usecols=np.arange(1,13))
        index = range(1979 , 2021)
        columns = range(1,13)
        meiv2 = pd.DataFrame(data = meiv2, columns = columns, index = index)
        meiv2 = meiv2.stack().reset_index()
        meiv2.columns = ['year', 'month', 'meiv2']
        meiv2['time'] = pd.to_datetime(meiv2.year*100+meiv2.month,format='%Y%m')
        meiv2 = meiv2.set_index('time').meiv2
        meiv2 = meiv2[meiv2>-80]
        meiv2 = xr.DataArray(meiv2)
        ds['meiv2'] = meiv2

    ds = xr.Dataset(ds)
    ds = ds.resample(time="1MS").mean()
    ds = ds.sel(time=slice("1979-01-01", "2021-06-10"))

    if 'seasonal' in temporal_resolution:
        ds = ds.resample(time="QS-DEC").mean()
    elif 'annual'  in temporal_resolution:
        ds = ds.resample(time="YS").mean()

    return ds

def normalise_indicies(ds):
    for index in ds:
        ind = ds[index].copy()
        ind = (ind - ind.mean()) 
        ind =  ind / ind.std()
        ds[index] = ind
    return ds

def plot_indicies(ds):
    """Plots the indicies in the dataset."""
    plt.style.use('stylesheets/timeseries.mplstyle')


    fig = plt.figure(figsize=[5,2*len(ds)])
    i = 0
    for index in tqdm(ds):
        data = ds[index].copy()
        data = data.dropna(dim='time')
        data_m, data_b, data_r_value, data_p_value, data_std_err = scipy.stats.linregress(data.time.values.astype(float), data)
        yfit = data_m * data.time.values.astype(float) + data_b
        ax = fig.add_subplot(len(ds),1,i+1)
        ax.plot(data.time, data)
        ax.plot(data.time, yfit,  color = '#177E89')
        print(f'{data_m:.2e}',max(data.values))
        ax.set_title(index)
        # ax.set_ylabel(index)
        i += 1
    # fig.suptitle("Index timeseries")
    plt.tight_layout()
    plt.savefig('images/week1/index_timeseries.pdf',bbox_inches='tight')
    plt.show()

def find_correlations(ds):
    variables = [v for v in ds.variables if v != 'time']
    correlations = pd.DataFrame(index = variables, columns = variables)
    pvalues = pd.DataFrame(index = variables, columns = variables)

    for x, y in tqdm(itertools.product(variables,variables)):
        x = ds[x].dropna(dim='time')
        y = ds[y].dropna(dim='time')

        times = list(set(set(x.time.values) & set(y.time.values)))

        x = x.sel(time=times)
        y = y.sel(time=times)

        corr, pval = scipy.stats.pearsonr(x,y)
        correlations.loc[x.name,y.name] = corr
        pvalues.loc[x.name,y.name] = pval
        # print(x.name,y.name,corr)

    return correlations, pvalues

def load_seaice(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/SIC/',seaice_source='nsidc'):
    seaice = xr.Dataset()
    for n, temp_res, temp_decomp, dt in itertools.product(resolutions, temporal_resolution, temporal_decomposition, detrend):
        output_folder = 'processed_data/SIC/'
        seaicename = f'{temp_decomp}_{temp_res}_{n}_{dt}'
        seaice[seaicename] = xr.open_dataset(output_folder + seaicename +'.nc')[seaicename]
    return seaice/250

def multiple_spatial_regression(seaice, ds):
    outputs = {}
    for seaicename in seaice:
        print(seaicename)
        sic = seaice[seaicename].copy()

        indicies = [ds[index].copy() for index in ds]
        index_names = [ind.name for ind in indicies]
        indicies = [ind.dropna(dim='time') for ind in indicies]
        # temporal averaging
        if 'seasonal' in seaicename:
            indicies = [ind.resample(time="QS-DEC").mean() for ind in indicies]
        elif 'annual' in seaicename:
            indicies = [ind.resample(time="YS").mean() for ind in indicies]
        times = list(set.intersection(set(sic.time.values), *(set(indicies[i].time.values)for i in range(len(indicies)))))
        sic = sic.sel(time=times).sortby('time')
        indicies = [ind.sel(time=times).sortby('time') for ind in indicies]

        new_indicies = xr.Dataset({v.name:v for v in indicies}).to_array(dim='variable')
        p0 = [0]*(len(indicies)+1)

        if len(indicies) == 4: linear_model = linear_model_4
        if len(indicies) == 5: linear_model = linear_model_5
        if len(indicies) == 6: linear_model = linear_model_6
        if len(indicies) == 4: apply_curvefit = apply_curvefit_4
        if len(indicies) == 5: apply_curvefit = apply_curvefit_5
        if len(indicies) == 6: apply_curvefit = apply_curvefit_6


        params = xr.apply_ufunc(apply_curvefit, 
                                linear_model, new_indicies, sic,
                                input_core_dims=[[], ['time','variable'], ['time']],
                                vectorize=True, # !Important!
                                dask='parallelized',
                                output_dtypes=[float]*(len(indicies)+1),
                                output_core_dims=[[]]*(len(indicies)+1)
                                )

        multiple_regressions = {index_names[i]:params[i] for i in range(len(params)-1)}
        multiple_regressions['error'] = params[-1]

        outputs[seaicename] = xr.Dataset(multiple_regressions)
    return outputs

def apply_curvefit_4(linear_model, newvariables, seaice):
    params, covariances = scipy.optimize.curve_fit(linear_model, newvariables.transpose(), seaice)
    a,b,c,d,e = params
    return a,b,c,d,e

def apply_curvefit_5(linear_model, newvariables, seaice):
    params, covariances = scipy.optimize.curve_fit(linear_model, newvariables.transpose(), seaice)
    a,b,c,d,e,f = params
    return a,b,c,d,e,f

def apply_curvefit_6(linear_model, newvariables, seaice):
    params, covariances = scipy.optimize.curve_fit(linear_model, newvariables.transpose(), seaice)
    a,b,c,d,e,f,g = params
    return a,b,c,d,e,f,g

def linear_model_4(x, a, b, c, d, e):
    return e + a*x[0] + b*x[1] + c*x[2] + d*x[3]

def linear_model_5(x, a, b, c, d, e, f):
    return f + a*x[0] + b*x[1] + c*x[2] + d*x[3] + e*x[4]

def linear_model_6(x, a, b, c, d, e, f, g):
    return g + a*x[0] + b*x[1] + c*x[2] + d*x[3] + e*x[4] + f*x[5]


def plot_coefficients(regression_results):
    for name,regression_result in regression_results.items():
        subindicies = [v for v in regression_result]
        N = len(subindicies)-2
        # if N%2 == 0: fig = plt.figure(figsize=(5/2 * N/2,5))
        # else:        
        fig = plt.figure(figsize=(5*N,5))
        max_ = max([regression_result[indexname].max() for indexname in subindicies[:-2]])
        min_ = min([regression_result[indexname].min() for indexname in subindicies[:-2]])
        divnorm = TwoSlopeNorm(vmin=min_, vcenter=0, vmax=max_)
        for i in range(N):
            data = regression_result[subindicies[i]]
            data = data.where(np.abs(data) != 0.0)
            # if N == 4:
            #     ax = fig.add_subplot(N//2,2,i+1, projection = ccrs.SouthPolarStereo())
            # else: 
            ax = fig.add_subplot(1,N,i+1, projection = ccrs.SouthPolarStereo())
            if 'x' in data.dims:
                ax.contourf(data.x,data.y,data.values, cmap = 'RdBu',norm = divnorm, transform = ccrs.SouthPolarStereo())
            else:
                ax.contourf(data.longitude,data.latitude,data.values, cmap = 'RdBu',norm = divnorm, transform = ccrs.PlateCarree())
            ax.set_title(subindicies[i])
            ax.coastlines()
        features = name.split('_')
        fig.suptitle(f'Regression coefficients')
        fig.subplots_adjust(right=0.95)
        cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(cm.ScalarMappable(norm=divnorm, cmap='RdBu'), cax=cbar_ax, shrink=0.88)
        cbar.set_label('Regression Coefficients [$\\frac{\%}{\sigma}$]')

        plt.savefig(f'images/week1/coefficients_{name}_'+'_'.join(subindicies)+'.pdf')



def contribution_to_trends(sic, prediction, indicies, ds, regression_results):

    data_m = prediction.sortby('time').polyfit(dim='time', deg=1) * 1e9*60*60*24*365
    seaice_gradient = sic.sortby('time').polyfit(dim='time', deg=1) * 1e9*60*60*24*365
    index_gradient = ds.sortby('time').polyfit(dim='time', deg=1) * 1e9*60*60*24*365
    for seaicename in sic:
        fig = plt.figure(figsize = (15,5))

        seaice_m = seaice_gradient[seaicename+'_polyfit_coefficients'].sel(degree=1)
        seaice_m = seaice_m.where(seaice_m !=0)
        max_ = seaice_m.max()
        min_ = seaice_m.min() 
        divnorm = TwoSlopeNorm(vmin=min_, vcenter=0, vmax=max_)
        ax = fig.add_subplot(131, projection = ccrs.SouthPolarStereo())
        # Plotting
        if 'x' in seaice_m.dims:
            contor = ax.contourf(seaice_m.x, seaice_m.y, seaice_m, cmap = 'RdBu', levels = 11, norm = divnorm, transform=ccrs.SouthPolarStereo())
        else:
            contor = ax.contourf(seaice_m.longitude, seaice_m.latitude, seaice_m, cmap = 'RdBu', levels = 11, norm = divnorm, transform=ccrs.PlateCarree())
        ax.coastlines()
        ax.set_axis_off()
        ax.set_title('Trend')
        cbar = plt.colorbar(contor)
        cbar.set_label('Trend (\% yr$^{-1}$)')
        features = seaicename.split('_')
        fig.suptitle(f'Seaice trends')
        
        data = regression_results[seaicename]['prediction'].polyfit(dim='time', deg=1).sel(degree=1).polyfit_coefficients * 1e9*60*60*24*365
        # print(data)
        data = data.where(abs(data) != 0.0)
        ax2 = fig.add_subplot(132, projection = ccrs.SouthPolarStereo())

        if 'x' in seaice_m.dims:
            contor = ax2.contourf(data.x, data.y, data.values, cmap = 'RdBu', levels = 11, norm = divnorm, transform=ccrs.SouthPolarStereo())
        else:
            contor = ax2.contourf(data.longitude, data.latitude, data, cmap = 'RdBu', levels = 11, norm = divnorm, transform=ccrs.PlateCarree())
        # contor = ax2.contourf(data.x, data.y, data.values, cmap = 'RdBu', levels = 11, norm = divnorm, transform=ccrs.SouthPolarStereo())
        ax2.coastlines()
        ax2.set_axis_off()
        cbar = plt.colorbar(contor)
        cbar.set_label('Trend (\% yr$^{-1}$)')
        ax2.set_title('Predicted Trend')
        

        residual = seaice_m - data
        residual = residual.where(abs(residual) != 0.0)
        ax = fig.add_subplot(133, projection = ccrs.SouthPolarStereo())

        if 'x' in seaice_m.dims:
            contor = ax.contourf(residual.x, residual.y, residual.values, cmap = 'RdBu', levels = 11, norm = divnorm, transform=ccrs.SouthPolarStereo())
        else:
            contor = ax.contourf(residual.longitude, residual.latitude, residual, cmap = 'RdBu', levels = 11, norm = divnorm, transform=ccrs.PlateCarree())
        # contor = ax.contourf(residual.x, residual.y, residual.values, cmap = 'RdBu', norm = divnorm, levels = 11, transform=ccrs.SouthPolarStereo())
        ax.coastlines()
        ax.set_axis_off()
        cbar = plt.colorbar(contor)
        cbar.set_label('Trend (\% yr$^{-1}$)')
        ax.set_title('Residual')
        plt.savefig(f'images/week1/Trend_Contribution_{seaicename}_'+'_'.join(indicies)+'.pdf')
        plt.show()


def plot_contribution_timeseries(sic,prediction, ds, regression_results):

    area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
    for seaicename in sic:
        if 'x' in sic.dims:
            seaice_timeseries = (sic[seaicename]*area).sum(dim=('x','y'))
            predicted_timeseries = (regression_results[seaicename].prediction*area).sum(dim=('x','y'))
            predicted_timeseries = predicted_timeseries.where(predicted_timeseries != 0)
        else:
            seaice_timeseries = (sic[seaicename]).mean(dim=('longitude','latitude'))
            predicted_timeseries = (regression_results[seaicename].prediction).mean(dim=('longitude','latitude'))
            predicted_timeseries = predicted_timeseries.where(predicted_timeseries != 0)

        seaice_m = seaice_timeseries.polyfit(dim='time', deg=1)
        sic_m = seaice_m.polyfit_coefficients.sel(degree=1).values
        b = seaice_m.polyfit_coefficients.sel(degree=0).values
        linear_sic = sic_m * seaice_timeseries.time.values.astype(float) + b

        predicted_m = predicted_timeseries.polyfit(dim='time', deg=1)
        pred_m = predicted_m.polyfit_coefficients.sel(degree=1).values
        b = predicted_m.polyfit_coefficients.sel(degree=0).values
        linear_predict = pred_m * seaice_timeseries.time.values.astype(float) + b


        subresults = regression_results[seaicename]
        indicies = [v for v in subresults][:-1]


        plt.style.use('stylesheets/timeseries.mplstyle')
        fig  = plt.figure()
        ln1 = plt.plot(sic.time, seaice_timeseries,color = '#1EA0AE', label = 'SIE')
        plt.plot(sic.time, linear_sic, color = '#1EA0AE')
        plt.axhline(0,alpha = 0.2)
        ln2 = plt.plot(predicted_timeseries.time, predicted_timeseries, color = '#BF160D', label='Prediction')
        plt.plot(seaice_timeseries.time, linear_predict, color = '#BF160D')
        
        lines = ln1 + ln2
        labels = [line.get_label() for line in lines]
        plt.legend(lines,labels,bbox_to_anchor=(0.99, -0.15), ncol = 3, loc = 'upper right')
        fig.suptitle('Predicted SIE')
        plt.ylabel('SIE [$km^2$]')
        plt.savefig(f'images/week1/Timeseries_{seaicename}_'+'_'.join(indicies)+'.pdf')
        plt.show()


def multiple_fast_regression(seaice, ds):
    outputs = {}
    for seaicename in seaice:
        if type(seaicename) == type(str()):
            print(seaicename)
            time.sleep(0.2)
            sic = seaice[seaicename].copy()
        else:
            sic = seaice.copy()

        indicies = [ds[index].copy() for index in ds]
        index_names = [ind.name for ind in indicies]
        indicies = [ind.stack(z=('y','x')).dropna(dim='z').dropna(dim='time').unstack() for ind in indicies]
        x = indicies[0].x.sortby('x')
        y = indicies[0].y.sortby('y')
        sic = sic.sel(x=x)
        sic = sic.sel(y=y)
        indicies = [ind.sel(x=x) for ind in indicies]
        indicies = [ind.sel(y=y) for ind in indicies]
        # temporal averaging
        # if 'seasonal' in seaicename:
        #     indicies = [ind.resample(time="QS-DEC").mean() for ind in indicies]
        # elif 'annual' in seaicename:
        #     indicies = [ind.resample(time="YS").mean() for ind in indicies]
        times = list(set.intersection(set(sic.time.values), *(set(indicies[i].time.values)for i in range(len(indicies)))))
        sic = sic.sel(time=times).sortby('time')
        indicies = [ind.sel(time=times).sortby('time') for ind in indicies]

        new_indicies = xr.Dataset({v.name:v for v in indicies}).to_array(dim='variable')
        
        params, yhat = fast_regression(new_indicies,sic)

        dims = [dim for dim in sic.dims if dim != 'time']

        multiple_regressions = {index_names[i]:xr.DataArray(data = params[i], dims=dims, coords = [sic[dims[0]], sic[dims[1]]]) for i in range(len(params)-1)}
        multiple_regressions['error'] = xr.DataArray(data = params[-1], dims=dims, coords = [sic[dims[0]], sic[dims[1]]])

        multiple_regressions['prediction'] = xr.DataArray(data =yhat, dims=('time',*dims), coords = [sic.time, sic[dims[0]], sic[dims[1]]])


        outputs[seaicename] = xr.Dataset(multiple_regressions)

    return outputs


from scipy.linalg import lstsq

def fast_regression(X,y):
    X = X.values
    newX = np.ones([X.shape[0]+1,*X.shape[1:]])
    newX[:-1,:] = X
    X = newX.transpose()
    y = y.values
    p = np.empty([X.shape[-1],*y.shape[1:]])


    print('Finding coefficients')

    if len(X.shape) == 2:
        X = X.transpose()
        X = np.repeat(X[:, np.newaxis, np.newaxis, :], y.shape[2], axis=2)
        X = np.repeat(X, y.shape[1], axis = 1)

    time.sleep(0.2)
    for i,j in tqdm(list(itertools.product(range(y.shape[1]), range(y.shape[2])))):
        if not np.isnan(X[j,i,:,:]).any():
            p[:,i,j] = lstsq(X[j,i,:,:], y[:,i,j].transpose())[0]
        else:
            p[:,i,j] = 0

    yhat = y.copy()
    print('Predicting SIC')
    time.sleep(0.2)
    yhat = np.einsum('jitn,nij->tij',X,p)
    return p, yhat


def individual_contribution_to_trends(regression_results, normalized, sic):
    index_gradient = normalized.sortby('time').polyfit(dim='time', deg=1) * 1e9*60*60*24*365
    
    for mode in regression_results.keys():
        regression_result = regression_results[mode]
        data = {}
        normalized = normalized.stack(z=('x','y')).dropna(dim='z').unstack().sortby('x').sortby('y')
        for index in normalized:
            array = np.einsum('xy,tyx->xyt',regression_result[index].values,normalized[index].values)
            dims = regression_result[index].dims
            data[index] = xr.DataArray(data = array, dims = [*dims,'time'], coords = (regression_result[dims[0]],regression_result[dims[1]],normalized.time))
    
    DATA = xr.Dataset(data).sortby('time').polyfit(dim='time', deg=1).sel(degree=1) * 1e9*60*60*24*365
    if 'x' in DATA.dims:
        area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
        timeseries = (xr.Dataset(data).sortby('time')*area).sum(dim=('x','y'))
    else:
        timeseries = xr.Dataset(data).sortby('time').mean(dim=('longitude', 'latitude'))
    for name,regression_result in regression_results.items():
        subindicies = [v for v in regression_result]
        N = len(subindicies)-2

        plt.style.use('stylesheets/contour.mplstyle')
        fig = plt.figure(figsize=(5*N,5))

        if 'x' in DATA.dims:
            DATA = DATA

        max_ = max([DATA[indexname+'_polyfit_coefficients'].max() for indexname in subindicies[:-2]])
        min_ = min([DATA[indexname+'_polyfit_coefficients'].min() for indexname in subindicies[:-2]])
        divnorm = TwoSlopeNorm(vmin=min_, vcenter=0, vmax=max_)
        for i in range(N):
            data = DATA[subindicies[i]+'_polyfit_coefficients']
            data = data.where(np.abs(data) != 0.000)
            # if N == 4:
            #     ax = fig.add_subplot(N//2,2,i+1, projection = ccrs.SouthPolarStereo())
            # else: 
            ax = fig.add_subplot(1,N,i+1, projection = ccrs.SouthPolarStereo())
            if 'x' in data.dims:
                ax.contourf(data.x,data.y,data.values, cmap = 'RdBu',norm = divnorm, transform = ccrs.SouthPolarStereo())
            else:
                ax.contourf(data.longitude,data.latitude,data.values, cmap = 'RdBu',norm = divnorm, transform = ccrs.PlateCarree())
            ax.set_title(subindicies[i])
            ax.coastlines()
        features = name.split('_')
        fig.suptitle(f'Regression contributions')
        fig.subplots_adjust(right=0.95)
        cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(cm.ScalarMappable(norm=divnorm, cmap='RdBu'), cax=cbar_ax, shrink=0.88)
        cbar.set_label('Regression Contributions [\% yr$^{-1}$]')

        plt.savefig(f'images/week1/contributions_{name}_'+'_'.join(subindicies)+'.pdf')
        plt.show()

    for seaicename in sic:
        if 'x' in sic.dims:
            seaice_timeseries = (sic[seaicename]*area).sum(dim=('x','y'))
            predicted_timeseries = (regression_results[seaicename].prediction*area).sum(dim=('x','y'))
            predicted_timeseries = predicted_timeseries.where(predicted_timeseries != 0)
        else:
            seaice_timeseries = (sic[seaicename]).mean(dim=('longitude','latitude'))
            predicted_timeseries = (regression_results[seaicename].prediction).mean(dim=('longitude','latitude'))
            predicted_timeseries = predicted_timeseries.where(predicted_timeseries != 0)

        seaice_m = seaice_timeseries.polyfit(dim='time', deg=1)
        sic_m = seaice_m.polyfit_coefficients.sel(degree=1).values
        b = seaice_m.polyfit_coefficients.sel(degree=0).values
        linear_sic = sic_m * seaice_timeseries.time.values.astype(float) + b

        predicted_m = predicted_timeseries.polyfit(dim='time', deg=1)
        pred_m = predicted_m.polyfit_coefficients.sel(degree=1).values
        b = predicted_m.polyfit_coefficients.sel(degree=0).values
        linear_predict = pred_m * seaice_timeseries.time.values.astype(float) + b


        subresults = regression_results[seaicename]
        indicies = [v for v in subresults][:-1]


        plt.style.use('stylesheets/timeseries.mplstyle')
        fig  = plt.figure()
        ln1 = plt.plot(sic.time, seaice_timeseries,color = '#1EA0AE', label = 'SIE', linewidth=0.5)
        plt.plot(sic.time, linear_sic, color = '#1EA0AE', linewidth=0.5)
        plt.axhline(0,alpha = 0.2)
        ln2 = plt.plot(predicted_timeseries.time, predicted_timeseries, color = '#BF160D', label='Prediction', linewidth=0.5)
        plt.plot(seaice_timeseries.time, linear_predict, color = '#BF160D', linewidth=0.5)
        
        lines = ln1 + ln2
        plt.plot([],[])
        plt.plot([],[])
        for i in range(N):
            index_timeseries = timeseries[subindicies[i]]
            lines += plt.plot(timeseries.time, index_timeseries, label = subindicies[i], linewidth=0.5)

            predicted_m = index_timeseries.polyfit(dim='time', deg=1)
            pred_m = predicted_m.polyfit_coefficients.sel(degree=1).values
            b = predicted_m.polyfit_coefficients.sel(degree=0).values
            linear_predict = pred_m * index_timeseries.time.values.astype(float) + b

            color = lines[-1].get_color()
            plt.plot(index_timeseries.time, linear_predict, color = color, linewidth=0.5)



        labels = [line.get_label() for line in lines]
        plt.legend(lines,labels,bbox_to_anchor=(0.99, -0.15), ncol = 3, loc = 'upper right')
        fig.suptitle('Predicted')
        plt.ylabel('SIE [$km^2$]')
        plt.savefig(f'images/week1/Timeseries_individual_{seaicename}_'+'_'.join(indicies)+'.pdf')
        plt.show()



def main(subindicies, resolutions, temporal_resolution,temporal_decomposition,detrend):
    sic = load_seaice(resolutions, temporal_resolution, temporal_decomposition, detrend)
    ds = load_indicies(subindicies, temporal_resolution)
    normalized = normalise_indicies(ds)[subindicies]
    regression_results = multiple_fast_regression(sic, normalized)
    plt.style.use('stylesheets/contour.mplstyle')
    plot_coefficients(regression_results)

    prediction = regression_results.copy()
    for mode in regression_results.keys():
        prediction[mode] = regression_results[mode]['prediction']
    prediction = xr.Dataset(prediction)
    plt.style.use('stylesheets/contour.mplstyle')
    contribution_to_trends(sic, prediction, subindicies, normalized, regression_results)
    plt.style.use('stylesheets/timeseries.mplstyle')
    plot_contribution_timeseries(sic,prediction, normalized, regression_results)
    plt.style.use('stylesheets/contour.mplstyle')
    individual_contribution_to_trends(regression_results, normalized, sic)

def main_stacked(subindicies_list, resolutions, temporal_resolution,temporal_decomposition,detrend):

    for subindicies in subindicies_list:
        main(subindicies, resolutions, temporal_resolution,temporal_decomposition,detrend)
        # main2(subindicies, resolutions, temporal_resolution,temporal_decomposition,detrend)

def get_stats(subindicies_list, resolutions, temporal_resolution,temporal_decomposition,detrend, variable='seaice'):
    cols = ['spatial_correlation','temporal_correlation','Predicted_Trend', 'Actual_Trend']
    IDs = ['_'.join(subindex) for subindex in subindicies_list]

    area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
    results = pd.DataFrame(columns = cols)
    for subindicies in subindicies_list:
        ID = '_'.join(subindicies)
        if variable == 'seaice':
            sic = load_seaice(resolutions, temporal_resolution, temporal_decomposition, detrend)
        else:
            sic = w2.load_variable(temporal_resolution[0],temporal_decomposition[0],detrend[0], variable)
            sic = xr.Dataset({variable:sic})
        ds = load_indicies(subindicies, temporal_resolution)
        normalized = normalise_indicies(ds)[subindicies]
        regression_results = multiple_fast_regression(sic, normalized)
        prediction = {key: regression_result.prediction for key, regression_result in regression_results.items()}

        # Spatial correlation
        prediction_trend = {key: p.polyfit(dim='time', deg=1).sel(degree=1) * 1e9*60*60*24*365 for key, p in prediction.items()}
        sic_trend        = sic.polyfit(dim='time', deg=1).sel(degree=1) * 1e9*60*60*24*365

        for seaicename in sic:
            seaice_trend = sic_trend[seaicename+'_polyfit_coefficients']
            predict_trend = prediction_trend[seaicename]['polyfit_coefficients']
            if variable == 'seaice':
                corr = xr.corr(seaice_trend, predict_trend, dim = ('x','y')).values
            else:
                corr = xr.corr(seaice_trend, predict_trend, dim = ('longitude','latitude')).values
            results.loc[ID+'_'+seaicename,'spatial_correlation'] = corr

            # Temporal Correlation
            if variable == 'seaice':
                seaice_timeseries = (sic[seaicename]*area).sum(dim=('x','y'))
                predicted_timeseries = (regression_results[seaicename].prediction*area).sum(dim=('x','y'))
            else:
                seaice_timeseries = (sic[seaicename]).mean(dim=('longitude','latitude'))
                predicted_timeseries = (regression_results[seaicename].prediction).mean(dim=('longitude','latitude'))
            predicted_timeseries = predicted_timeseries.where(predicted_timeseries != 0)
            corr = xr.corr(seaice_timeseries, predicted_timeseries).values
            results.loc[ID+'_'+seaicename,'temporal_correlation'] = corr

            seaice_m = seaice_timeseries.polyfit(dim='time', deg=1)* 1e9*60*60*24*365
            sic_m = seaice_m.polyfit_coefficients.sel(degree=1).values

            results.loc[ID+'_'+seaicename,'Actual_Trend'] = sic_m

            predicted_m = predicted_timeseries.polyfit(dim='time', deg=1)* 1e9*60*60*24*365
            pred_m = predicted_m.polyfit_coefficients.sel(degree=1).values
            
            results.loc[ID+'_'+seaicename,'Predicted_Trend'] = pred_m

    return results

def subgradients(subindicies_list, resolutions, temporal_resolution,temporal_decomposition,detrend):
    cols = []
    for subindex in subindicies_list: cols = set(cols) | set(subindex)
    cols = list(cols) + ['Actual_Trend', 'Predicted_Trend']
    IDs = ['_'.join(subindex) for subindex in subindicies_list]

    area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
    results = pd.DataFrame(columns = cols)
    for subindicies in subindicies_list:
        ID = '_'.join(subindicies)
        sic = load_seaice(resolutions, temporal_resolution, temporal_decomposition, detrend)
        ds = load_indicies(subindicies, temporal_resolution)
        normalized = normalise_indicies(ds)[subindicies]
        regression_results = multiple_fast_regression(sic, normalized)
        prediction = {key: regression_result.prediction for key, regression_result in regression_results.items()}

        # Spatial correlation
        prediction_trend = {key: p.polyfit(dim='time', deg=1).sel(degree=1) * 1e9*60*60*24*365 for key, p in prediction.items()}
        sic_trend        = sic.polyfit(dim='time', deg=1).sel(degree=1) * 1e9*60*60*24*365

        for seaicename in sic:
            seaice_trend = sic_trend[seaicename+'_polyfit_coefficients']
            predict_trend = prediction_trend[seaicename]['polyfit_coefficients']
            corr = xr.corr(seaice_trend, predict_trend, dim = ('x','y')).values

            # Temporal Correlation
            seaice_timeseries = (sic[seaicename]*area).sum(dim=('x','y'))
            predicted_timeseries = (regression_results[seaicename].prediction*area).sum(dim=('x','y'))
            predicted_timeseries = predicted_timeseries.where(predicted_timeseries != 0)
            corr = xr.corr(seaice_timeseries, predicted_timeseries).values

            seaice_trend = sic_trend[seaicename+'_polyfit_coefficients']
            predict_trend = prediction_trend[seaicename]['polyfit_coefficients']

            seaice_m = seaice_timeseries.polyfit(dim='time', deg=1)* 1e9*60*60*24*365
            sic_m = seaice_m.polyfit_coefficients.sel(degree=1).values
            results.loc[ID+'_'+seaicename,'Actual_Trend'] = sic_m

            predicted_m = predicted_timeseries.polyfit(dim='time', deg=1)* 1e9*60*60*24*365
            pred_m = predicted_m.polyfit_coefficients.sel(degree=1).values
            results.loc[ID+'_'+seaicename,'Predicted_Trend'] = pred_m

            for mode in regression_results.keys():
                if 'seasonal' in mode:
                    normalized = normalized.resample(time="QS-DEC").mean()
                elif 'annual' in mode:
                    normalized = normalized.resample(time="YS").mean()
                regression_result = regression_results[mode]
                data = {}
                for index in normalized:
                    array = np.einsum('xy,t->xyt',regression_result[index].values,normalized[index].values)
                    data[index] = xr.DataArray(data = array, dims = ('y','x','time'), coords = (regression_result.y,regression_result.x,normalized.time))
            area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
            timeseries = (xr.Dataset(data).sortby('time')*area).sum(dim=('x','y'))

            for index in timeseries:
                index_timeseries = timeseries[index].polyfit(dim='time', deg=1)* 1e9*60*60*24*365
                index_m = index_timeseries.polyfit_coefficients.sel(degree=1).values
                results.loc[ID+'_'+seaicename,index] = index_m

    return results


def individual_main(subindicies, resolutions, temporal_resolution,temporal_decomposition,detrend):
    sic = load_seaice(resolutions, temporal_resolution, temporal_decomposition, detrend)
    ds = load_indicies(subindicies, temporal_resolution)
    normalized = normalise_indicies(ds)[subindicies]
    for index in normalized:
        normalized_i = xr.Dataset({index:normalized[index]})
        regression_results = multiple_fast_regression(sic, normalized_i)
        plt.style.use('stylesheets/contour.mplstyle')
        plot_coefficients(regression_results)

        prediction = regression_results.copy()
        for mode in regression_results.keys():
            prediction[mode] = regression_results[mode]['prediction']
        prediction = xr.Dataset(prediction)
        plt.style.use('stylesheets/contour.mplstyle')
        contribution_to_trends(sic, prediction, subindicies, normalized_i, regression_results)
        plt.style.use('stylesheets/timeseries.mplstyle')
        plot_contribution_timeseries(sic,prediction, normalized_i, regression_results)
        plt.style.use('stylesheets/contour.mplstyle')
        individual_contribution_to_trends(regression_results, normalized_i, sic)

def individual_main_stacked(subindicies_list, resolutions, temporal_resolution,temporal_decomposition,detrend):

    for subindicies in subindicies_list:
        main(subindicies, resolutions, temporal_resolution,temporal_decomposition,detrend)
        individual_main(subindicies, resolutions, temporal_resolution,temporal_decomposition,detrend)