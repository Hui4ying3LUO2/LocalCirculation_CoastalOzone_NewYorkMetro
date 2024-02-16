import os

os.environ["PROJ_LIB"] = 'C:\\Users\\Huiying\\Anaconda3\\envs\\luopy\\Library\\share\\basemap'

from math import cos, sin, asin, sqrt
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from netCDF4 import Dataset
from sklearn.cluster import KMeans
import random
import joblib
from mpl_toolkits.basemap import Basemap
from math import cos, sin, asin, sqrt
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from netCDF4 import Dataset
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

############# Year control

yearlist = ['2017', '2018', '2019']
days_n = 92 * len(yearlist)
d_name='NYCLI'
Metid=['QUEE']
outpath='SIxHotSBW/'
############# Figure configurations
cols = ['#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#a6cee3', '#1f78b4', '#b2df8a',
        '#33a02c', '#e9a3c9', '#c51b7d', '#998ec3', '#542788']  # light, dark, light, dark...
ttfont = 20
tkfont = 20

############ Met Cluster

Windname = ['Sea Breeze', 'Oscillation', 'Southerly', 'Westerly']
fn = '2WindCluster\kmeans161\QUEEkmeans_norm1__6_4CLusterAssign.txt'
with open(fn, 'r') as f:
    text = f.read()
    WindC = text.split()

Solarname = ['Hot', 'Moderate',
             'Cool']  # ['Clear','Cloudy','Overcast']#['Clear','LightClouds','PMClouds','Cloudy','Overcast']
fn = '3TSolarCluster\TCLusterAssignQUEEkmeans_norm0__24_3.txt'
with open(fn, 'r') as f:
    text = f.read()
    SolarC = text.split()

fn = '3TSolarCluster\SCLusterDateQUEEkmeans_norm0__11_3.txt'
with open(fn, 'r') as f:
    text = f.read()
    Metdate = text.split()

# MESO
Met_T = []
for yr in yearlist:
    inpath='C:\Projects\WRF_SUNY\OBS\\NYSM_Standard_hourlyEST\\MESONET_surfacehourly_' + yr + '0601_' + yr + '0831EST'+d_name+'.nc'
    if os.path.isfile(inpath):
        ds = Dataset(inpath)
        if yr==yearlist[0]:
            site_id_meso = ds.variables['site_id'][:] # 21 sites
            site_name_meso = ds.variables['site_name'][:]
            site_lat_meso = ds.variables['site_lat'][:]
            site_lon_meso = ds.variables['site_lon'][:]

            DS_Td_meso = ds.variables['Td'][:]
            DS_P_meso = ds.variables['PS'][:]
            DS_T2_meso = ds.variables['T2'][:]
            DS_wd_meso = ds.variables['WS'][:]
            DS_wddir_meso = ds.variables['Wdir'][:]
            DS_U_meso = ds.variables['U'][:]
            DS_V_meso = ds.variables['V'][:]
            DS_solar_meso = ds.variables['solar'][:]
        else:
            DS_Td_meso = np.concatenate((DS_Td_meso, ds.variables['Td'][:]))
            DS_P_meso = np.concatenate((DS_P_meso,ds.variables['PS'][:]))
            DS_T2_meso = np.concatenate((DS_T2_meso,ds.variables['T2'][:]))
            DS_wd_meso = np.concatenate((DS_wd_meso,ds.variables['WS'][:]))
            DS_wddir_meso = np.concatenate((DS_wddir_meso,ds.variables['Wdir'][:]))
            DS_U_meso = np.concatenate((DS_U_meso,ds.variables['U'][:]))
            DS_V_meso = np.concatenate((DS_V_meso,ds.variables['V'][:]))
            DS_solar_meso = np.concatenate((DS_solar_meso, ds.variables['solar'][:]))
        for i in range(len(ds.variables['WS'])):
            Met_T.append(datetime.strptime(ds.variables['time'][i], '%Y-%m-%d%H:%M'))


DS_U=DS_U_meso
DS_V=DS_V_meso
DS_wd=DS_wd_meso
DS_wddir=DS_wddir_meso
DS_P=DS_P_meso
DS_T2=DS_T2_meso
DS_solar=DS_solar_meso

siteid=site_id_meso
sitename =site_name_meso
site_lat = site_lat_meso
site_lon = site_lon_meso

M_Time = np.array(Met_T).reshape(days_n, 24)
M_Hour = np.arange(24)
M_Date = M_Time[:, 0]

M_datevalid = np.zeros([len(siteid), days_n])
M_datevalid[:] = 1
clustern=2
para_n=1
clusterm = 'kmeans'


if os.path.exists(outpath):
    print("Folder exists! Overwriting...")
else:
    os.mkdir(outpath)

M_y = np.zeros([len(siteid), days_n])
M_y[:] = np.nan
Mgp = np.zeros([len(siteid), days_n])  # assign cluster
Mgp[:] = np.nan

for s in range(len(siteid)):
    if siteid[s] == 'WANT':

        M_para = np.zeros([days_n, para_n]) # calculated parameter
        M_para[:] = np.nan
        M_pn = np.zeros([days_n, para_n]) # normalized
        M_pn[:] = np.nan

        U_site = np.array(DS_U[:, s])
        V_site = np.array(DS_V[:, s])
        Wdir_site = np.array(DS_wddir[:, s])
        WS_site = np.array(DS_wd[:, s])
        PS_site = np.array(DS_P[:, s])
        TEMP_site = np.array(DS_T2[:, s])
        solar_site = np.array(DS_solar[:, s])

        Mt_U = U_site.reshape(days_n, 24)
        Mt_V = V_site.reshape(days_n, 24)
        Mt_WD = Wdir_site.reshape(days_n, 24)
        Mt_WS = WS_site.reshape(days_n, 24)
        Mt_PS = PS_site.reshape(days_n, 24)
        Mt_TEMP = TEMP_site.reshape(days_n, 24)
        Mt_solar = solar_site.reshape(days_n, 24)

        # linear fit to fill in gaps
        U_sitep = pd.DataFrame(data=U_site)
        U_sitefit = U_sitep.interpolate()
        M_U = U_sitefit.to_numpy().reshape(days_n, 24)

        V_sitep = pd.DataFrame(data=V_site)
        V_sitefit = V_sitep.interpolate()
        M_V = V_sitefit.to_numpy().reshape(days_n, 24)

        WD_sitep = pd.DataFrame(data=Wdir_site)
        WD_sitefit = WD_sitep.interpolate()
        M_WD = WD_sitefit.to_numpy().reshape(days_n, 24)

        WS_sitep = pd.DataFrame(data=WS_site)
        WS_sitefit = WS_sitep.interpolate()
        M_WS = WS_sitefit.to_numpy().reshape(days_n, 24)

        PS_sitep = pd.DataFrame(data=PS_site)
        PS_sitefit = PS_sitep.interpolate()
        M_PS = PS_sitefit.to_numpy().reshape(days_n, 24)

        TEMP_sitep = pd.DataFrame(data=TEMP_site)
        TEMP_sitefit = TEMP_sitep.interpolate()
        M_TEMP = TEMP_sitefit.to_numpy().reshape(days_n, 24)

        #
        for sid in [0]:
            for wid in [3]:
                indices = [i for i, s in enumerate(SolarC) if str(float(sid)) in s]
                indices2 = [i for i, s in enumerate(WindC) if str(float(wid)) in s]
                dlist = list(set(indices2).intersection(indices))
        for t in range(len(M_Date)):
            if t not in dlist:
                M_datevalid[s, t] = 0
                M_V[t, :] = np.nan
                M_U[t, :] = np.nan
                M_WD[t, :] = np.nan
                M_WS[t, :] = np.nan
                M_PS[t, :] = np.nan
                M_TEMP[t, :] = np.nan


        if sum(M_datevalid[s, :]) > 10:  # more than 30 valid days sitem_id[s]=='BKLN': #s in range(len(sitem_id)):##

            # features calculation and normalization
            # A: original, normalized 7

            for t in range(len(M_Date)):
                if M_datevalid[s, t] == 1:
                    M_para[t, 0] = np.nanmean(M_V[t, 13:17])  # 3:7


            # kmean cluster
            # if clusterm == 'kmeans':
            #     kmeans = KMeans(n_clusters=clustern, init='k-means++', max_iter=300, n_init=10, random_state=0)
            #     pred_y = kmeans.fit_predict(M_para[M_datevalid[s, :] == 1, :])

            pred_y=M_para[M_datevalid[s, :] == 1, :]>0

            M_y[s, M_datevalid[s, :] == 1] = pred_y.reshape(len(dlist))

            # assigning groups
            #
            Ul = np.zeros(clustern)
            Ul[:] = np.nan
            g_ind = np.zeros(clustern)
            g_ind[:] = np.nan

            Cluname=['W_sb','W_w']

            # for i in range(clustern):
            #     g_ind[i]=i

            for g in range(clustern):
                Ug = M_V[np.where(M_y[s, :] == g), :]
                Ul[g] = np.nanmean(Ug)

            Ul = np.argsort(Ul)[::-1]  # desc
            # w
            g_ind[0]=Ul[0]# large v in the marning - S
            g_ind[1]=Ul[1]# - v in the marning - O

            fig= plt.figure(figsize=(10, 5))
            for g in range(clustern):
                Ug = Mt_U[np.where(M_y[s, :] == g_ind[g]), :]
                Vg = Mt_V[np.where(M_y[s, :] == g_ind[g]), :]
                solarg = Mt_solar[np.where(M_y[s, :] == g_ind[g]), :]
                Ul = np.nanmean(Ug, axis=1).reshape(24)
                Vl = np.nanmean(Vg, axis=1).reshape(24)
                # C = np.nanmean(solarg, axis=1).reshape(24)
                C = (Ul * Ul + Vl * Vl)
                for ci in range(len(C)):
                    C[ci] = sqrt(C[ci])
                qq = plt.quiver(M_Hour, g, Ul, Vl, C, cmap=plt.cm.jet, scale=35)
                plt.clim(0, 5)
                # str(Ug.shape[1])
            # plt.title('WANT Wind Subsets',
            #           fontsize=ttfont, fontweight='bold')
            plt.xlabel('EST', fontsize=tkfont)
            # plt.legend(fontsize="xx-large")
            cbar = plt.colorbar(qq, cmap=plt.cm.jet, fraction=0.04, pad=0.03)
            cbar.ax.tick_params(labelsize=tkfont)
            cbar.set_label('Wind Speed (m/s)', fontsize=20)

            plt.tick_params(labelsize=tkfont)
            plt.ylim([-0.5, clustern-0.5])
            plt.xticks(np.arange(0, 24, 4),
                       ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'])
            plt.yticks(np.linspace(0, clustern - 1, num=clustern), Cluname)
            plt.subplots_adjust(bottom=0.19)

            fig.savefig(
                outpath + '/ClusterMean.png', dpi=600)
            fig.savefig(
                outpath + '/ClusterMean.pdf', dpi=600)
            plt.close()



            for g in range(2):
                Mgp[s, np.where(M_y[s, :] == g_ind[g])] = g  # assigned

            with open(outpath + '/SubQUEEW_WANTCLusterAssign.txt', 'w') as testfile:
                for row in Mgp[s, :]:
                    testfile.write(str(row) + '\n')

            with open(outpath + '/SubQUEEW_WANTCLusterOrig.txt', 'w') as testfile:
                for row in M_y[s, :]:
                    testfile.write(str(row) + '\n')

            with open(outpath + '/SubQUEEW_WANTCLusterDate.txt', 'w') as testfile:
                for row in M_Date:
                    testfile.write(row.strftime('%Y-%m-%d') + '\n')





