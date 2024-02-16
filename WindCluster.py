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
from windrose import WindroseAxes
import glob
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin, asin, sqrt
import pandas as pd
from datetime import datetime
from netCDF4 import Dataset
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
from wrf import to_np, get_basemap, latlon_coords, getvar, ALL_TIMES
from sklearn_extra.cluster import KMedoids

###########

clustern = 4
datam='single' # region
# cluster_M=['kmeans','kmeans','kmeans','kmeans','kmedoids','kmedoids','kmedoids','kmedoids']
# Norm_YN=[1,1,0,0,1,1,0,0]
ratio=1 # ratio for morning and afternoon wind
cluster_M=['kmeans']
Norm_YN=[1]
para_N=[6]
metlist=['QUEE']
#metlist=['SDHN4','ROBN4','QPTR1','MHRN6','KPTN6','BRHC3','44069','44065','44040','44039','WANT','STON','STAT','SOUT','QUEE','MANH','BRON','BKLN','FOK','HWV','NYC','MTP','FRG','TEB','CDW','BDR']

outpath='2WindCluster/'+cluster_M[0]+str(Norm_YN[0])+str(para_N[0])+str(ratio)+'/'
yearlist=['2017','2018','2019']
days_n = 92*len(yearlist)

ttfont = 20
tkfont = 20
#############
def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())

######### Input loading
# Met: asos, buoy, aqs,mesonet

Met_T = []# use asos 2015-2019
#
# # ASOS: check LST
# for yr in yearlist:
#     inpath='C:\Projects\WRF_SUNY\OBS\ASOS\ASOS_surfacehourly_' + yr + '0601_' + yr + '0831LSTNYCLI.nc'
#     if os.path.isfile(inpath):
#         ds = Dataset(inpath)
#         if yr=='2015':
#             site_id_asos = ds.variables['site_id'][:]
#             site_name_asos = ds.variables['site_county'][:]
#             site_lat_asos = ds.variables['site_lat'][:]
#             site_lon_asos = ds.variables['site_lon'][:]
#
#             DS_P_asos = ds.variables['PS'][:]
#             DS_T2_asos = ds.variables['Temp'][:]
#             DS_wd_asos = ds.variables['WS'][:]
#             DS_wddir_asos = ds.variables['Wdir'][:]
#             DS_U_asos = ds.variables['U'][:]
#             DS_V_asos = ds.variables['V'][:]
#         else:
#             DS_P_asos = np.concatenate((DS_P_asos,ds.variables['PS'][:]))
#             DS_T2_asos = np.concatenate((DS_T2_asos,ds.variables['Temp'][:]))
#             DS_wd_asos = np.concatenate((DS_wd_asos,ds.variables['WS'][:]))
#             DS_wddir_asos = np.concatenate((DS_wddir_asos,ds.variables['Wdir'][:]))
#             DS_U_asos = np.concatenate((DS_U_asos,ds.variables['U'][:]))
#             DS_V_asos = np.concatenate((DS_V_asos,ds.variables['V'][:]))
#         for i in range(len(ds.variables['WS'])):
#             Met_T.append(datetime.strptime(ds.variables['time'][i], '%Y-%m-%d%H:%M'))
#
# # buoy
# for yr in yearlist:
#     inpath='C:\Projects\WRF_SUNY\OBS\\buoy\\buoy_surfacehourly_' + yr + '0601_' + yr + '0831ESTNYCLI.nc'
#     if os.path.isfile(inpath):
#         ds = Dataset(inpath)
#         if yr=='2015':
#             site_id_buoy = ds.variables['site_id'][:]
#             site_name_buoy = ds.variables['site_name'][:]
#             site_lat_buoy = ds.variables['site_lat'][:]
#             site_lon_buoy = ds.variables['site_lon'][:]
#
#             DS_P_buoy = ds.variables['PS'][:]
#             DS_T2_buoy = ds.variables['Temp'][:]
#             DS_wd_buoy = ds.variables['WS'][:]
#             DS_wddir_buoy = ds.variables['Wdir'][:]
#             DS_U_buoy = ds.variables['U'][:]
#             DS_V_buoy = ds.variables['V'][:]
#         else:
#             DS_P_buoy = np.concatenate((DS_P_buoy,ds.variables['PS'][:]))
#             DS_T2_buoy = np.concatenate((DS_T2_buoy,ds.variables['Temp'][:]))
#             DS_wd_buoy = np.concatenate((DS_wd_buoy,ds.variables['WS'][:]))
#             DS_wddir_buoy = np.concatenate((DS_wddir_buoy,ds.variables['Wdir'][:]))
#             DS_U_buoy = np.concatenate((DS_U_buoy,ds.variables['U'][:]))
#             DS_V_buoy = np.concatenate((DS_V_buoy,ds.variables['V'][:]))
#
# # AQS
# for yr in yearlist:
#     inpath='C:\Projects\WRF_SUNY\OBS\AQS\AQS_hourlymet_' + yr + '0601_' + yr + '0831ESTNYCLI.nc'
#     if os.path.isfile(inpath):
#         ds = Dataset(inpath)
#         if yr=='2015':
#             site_id_aqs = ds.variables['site_id'][:]
#             site_name_aqs = ds.variables['site_county'][:]
#             site_lat_aqs = ds.variables['site_lat'][:]
#             site_lon_aqs = ds.variables['site_lon'][:]
#
#             DS_P_aqs = ds.variables['PRESS'][:]
#             DS_T2_aqs = ds.variables['TEMP'][:]
#             DS_wd_aqs = ds.variables['WS'][:]
#             DS_wddir_aqs = ds.variables['Wdir'][:]
#             DS_U_aqs = ds.variables['U'][:]
#             DS_V_aqs = ds.variables['V'][:]
#         else:
#             DS_P_aqs = np.concatenate((DS_P_aqs,ds.variables['PRESS'][:]))
#             DS_T2_aqs = np.concatenate((DS_T2_aqs,ds.variables['TEMP'][:]))
#             DS_wd_aqs = np.concatenate((DS_wd_aqs,ds.variables['WS'][:]))
#             DS_wddir_aqs = np.concatenate((DS_wddir_aqs,ds.variables['Wdir'][:]))
#             DS_U_aqs = np.concatenate((DS_U_aqs,ds.variables['U'][:]))
#             DS_V_aqs = np.concatenate((DS_V_aqs,ds.variables['V'][:]))

# MESO
for yr in yearlist:
    inpath='C:\Projects\WRF_SUNY\OBS\\NYSM_Standard_hourlyEST\\MESONET_surfacehourly_' + yr + '0601_' + yr + '0831ESTNYCLI.nc'
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
            DS_solar_meso = np.concatenate((DS_solar_meso,ds.variables['solar'][:]))
        for i in range(len(ds.variables['WS'])):
            Met_T.append(datetime.strptime(ds.variables['time'][i], '%Y-%m-%d%H:%M'))


# an_array = np.empty((3*2208,len(site_id_meso)))
# an_array[:] = np.NaN
#
# DS_Td_meso = np.concatenate((an_array,DS_Td_meso),axis=0)
# DS_P_meso = np.concatenate((an_array,DS_P_meso),axis=0)
# DS_T2_meso = np.concatenate((an_array,DS_T2_meso),axis=0)
# DS_wd_meso = np.concatenate((an_array,DS_wd_meso),axis=0)
# DS_wddir_meso = np.concatenate((an_array,DS_wddir_meso),axis=0)
# DS_U_meso = np.concatenate((an_array,DS_U_meso),axis=0)
# DS_V_meso = np.concatenate((an_array,DS_V_meso),axis=0)
#
# ## combine all
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



# DS_U=np.concatenate((DS_U_asos,DS_U_aqs,DS_U_meso,DS_U_buoy),axis=1)
# DS_V=np.concatenate((DS_V_asos,DS_V_aqs,DS_V_meso,DS_V_buoy),axis=1)
# DS_wd=np.concatenate((DS_wd_asos,DS_wd_aqs,DS_wd_meso,DS_wd_buoy),axis=1)
# DS_wddir=np.concatenate((DS_wddir_asos,DS_wddir_aqs,DS_wddir_meso,DS_wddir_buoy),axis=1)
# DS_P=np.concatenate((DS_P_asos,DS_P_aqs,DS_P_meso,DS_P_buoy),axis=1)
# DS_T2=np.concatenate((DS_T2_asos,DS_T2_aqs,DS_T2_meso,DS_T2_buoy),axis=1)
#
# siteid=np.concatenate((site_id_asos,site_id_aqs,site_id_meso,site_id_buoy))
# sitename = np.concatenate((site_name_asos,site_name_aqs,site_id_meso,site_name_buoy))
# site_lat = np.concatenate((site_lat_asos,site_lat_aqs,site_lat_meso,site_lat_buoy))
# site_lon = np.concatenate((site_lon_asos,site_lon_aqs,site_lon_meso,site_lon_buoy))

# M: for model
M_Time = np.array(Met_T).reshape(days_n, 24)
M_Hour = np.arange(24)
M_Date = M_Time[:, 0]
# for d in range(days_n):
#     M_Date[d]=M_Date[d].date()
M_datevalid = np.zeros([len(siteid), days_n])
M_datevalid[:] = 1

#metlist=np.concatenate((metlist,site_id_aqs))
for pppp in para_N:

    for mid in range(len(cluster_M)):

        clusterm = cluster_M[mid]  # kmedoids'# kmeans
        normm = Norm_YN[mid]  #
        para_n =  pppp#


        if os.path.exists(outpath):
            print("Folder exists! Overwriting...")
        else:
            os.mkdir(outpath)

        M_y = np.zeros([len(siteid), days_n])
        M_y[:] = np.nan
        Mgp = np.zeros([len(siteid), days_n])  # assign cluster
        Mgp[:] = np.nan

        for s in range(len(siteid)):
            if siteid[s] in metlist:

                fig, ax = plt.subplots(figsize=(8, 6))
                bm = Basemap(llcrnrlon=-75, llcrnrlat=39.5,
                             urcrnrlon=-71, urcrnrlat=42.0,
                             resolution='h')

                bm.drawcoastlines(linewidth=0.25)
                bm.drawstates(linewidth=0.25)
                bm.drawcountries(linewidth=0.5)

                cs = bm.scatter(site_lon, site_lat, s=60, facecolor='0.5', linewidth=0.25)

                cs = bm.scatter(site_lon[s], site_lat[s], s=220, marker=(5, 1), facecolor='r')
                plt.text(-73.5, 39.8, sitename[s] + ' ' + siteid[s],fontweight='bold',
                         fontsize=ttfont)

                fig.savefig(outpath + '/' + str(siteid[s])  + '_loc.png', dpi=300,
                            bbox_inches='tight')


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

                # mark days not satisfy 90% data coverage, > 2days; set data to 0 for those days
                for t in range(len(M_Date)):
                    if sum(np.isnan(Mt_V[t, :])) > 2:# 11 12 13 EST or np.nanmean(Mt_solar[t, 11:14]) < 700
                        M_datevalid[s, t] = 0
                        M_V[t, :] = np.nan
                        M_U[t, :] = np.nan
                        M_WD[t, :] = np.nan
                        M_WS[t, :] = np.nan
                        M_PS[t, :] = np.nan
                        M_TEMP[t, :] = np.nan


                if sum(M_datevalid[s, :]) > 30:  # more than 30 valid days sitem_id[s]=='BKLN': #s in range(len(sitem_id)):##

                    # features calculation and normalization
                    # A: original, normalized 7
                    if para_n == 7:
                        for t in range(len(M_Date)):
                            if M_datevalid[s, t] == 1:
                                M_para[t, 0] = np.mean(M_U[t, 4:7])  # 3:7
                                M_para[t, 1] = np.mean(M_V[t, 4:7])
                                M_para[t, 2] = np.mean(M_U[t, 14:17])  # 14:18
                                M_para[t, 3] = np.mean(M_V[t, 14:17])
                                U24 = sum(M_U[t, :])
                                V24 = sum(M_V[t, :])
                                if U24 > 0:
                                    theta = np.pi / 2 - np.arctan(V24 / U24)
                                else:
                                    theta = np.pi / 2 * 3 - np.arctan(V24 / U24)
                                L = sqrt(U24 * U24 + V24 * V24)
                                S = 0
                                for h in range(24):
                                    S = S + sqrt(M_U[t, h] * M_U[t, h] + M_V[t, h] * M_V[t, h])
                                M_para[t, 5] = cos(theta)
                                M_para[t, 6] = sin(theta)
                                M_para[t, 4] = L / S

                        for f in range(7):
                            df = pd.DataFrame(data=M_para[:, f])
                            M_pn[:, f] = df.apply(norm_to_zero_one).to_numpy().reshape(days_n)
                        tt = '7'

                    elif para_n == 4:
                        for t in range(len(M_Date)):
                            if M_datevalid[s, t] == 1:
                                M_para[t, 0] = np.mean(M_U[t, 4:7])  # 3:7
                                M_para[t, 1] = np.mean(M_V[t, 4:7])
                                M_para[t, 2] = np.mean(M_U[t, 14:17])  # 14:18
                                M_para[t, 3] = np.mean(M_V[t, 14:17])
                        for f in range(4):
                            df = pd.DataFrame(data=M_para[:, f])
                            M_pn[:, f] = df.apply(norm_to_zero_one).to_numpy().reshape(days_n)
                        tt = '4'

                    elif para_n == 3:
                        for t in range(len(M_Date)):
                            if M_datevalid[s, t] == 1:

                                M_para[t, 0] = np.mean(M_WS[t, 4:7])  # small
                                M_para[t, 1] = np.mean(M_WS[t, 4:7]) / np.mean(M_WS[t, 14:17])  # wind speed ratio large
                                M_para[t, 2] = np.mean([abs(M_WD[t, 14] - 180), abs(M_WD[t, 15] - 180),
                                                        abs(M_WD[t, 16] - 180)])  # afternoon wind dir close to 180
                        for f in range(3):
                            df = pd.DataFrame(data=M_para[:, f])
                            M_pn[:, f] = df.apply(norm_to_zero_one).to_numpy().reshape(days_n)
                        tt = '3'
                    elif para_n == 5:
                        for t in range(len(M_Date)):
                            if M_datevalid[s, t] == 1:
                                M_para[t, 0] = np.mean(M_U[t, 4:7])  # 3:7
                                M_para[t, 1] = np.mean(M_V[t, 4:7])
                                M_para[t, 2] = np.mean(M_U[t, 14:17])  # 14:18
                                M_para[t, 3] = np.mean(M_V[t, 14:17])
                                M_para[t, 4] = np.mean(M_WS[t, 4:7]) #

                        for f in range(5):
                            df = pd.DataFrame(data=M_para[:, f])
                            M_pn[:, f] = df.apply(norm_to_zero_one).to_numpy().reshape(days_n)
                        M_pn[:, 4]=M_pn[:, 4]* ratio
                        tt = '5'

                    elif para_n == 6:
                        for t in range(len(M_Date)):
                            if M_datevalid[s, t] == 1:
                                M_para[t, 0] = np.mean(M_U[t, 4:7])  # 3:7
                                M_para[t, 1] = np.mean(M_V[t, 4:7])
                                M_para[t, 2] = np.mean(M_U[t, 14:17])  # 14:18
                                M_para[t, 3] = np.mean(M_V[t, 14:17])
                                M_para[t, 4] = np.mean(M_WS[t, 4:7]) / np.mean(M_WS[t, 14:17])  # wind speed ratio large
                                M_para[t, 5] = np.mean(M_WS[t, 4:7])  #
                        for f in range(6):
                            df = pd.DataFrame(data=M_para[:, f])
                            M_pn[:, f] = df.apply(norm_to_zero_one).to_numpy().reshape(days_n)
                        M_pn[:, 4] = M_pn[:, 4] * ratio
                        tt = '6'


                    # kmean cluster: find k
                    # wcss = []
                    # for i in range(1, 11):
                    #     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
                    #     kmeans.fit(M_O3[M_datevalid[s,:]==1,:])
                    #     wcss.append(kmeans.inertia_)
                    #
                    # fig = plt.figure(figsize=(12, 9))
                    # plt.plot(range(1, 11),wcss,'+-')
                    # plt.title(sitename[s]+', Elbow Method', fontsize=ttfont, fontweight='bold')
                    # plt.xlabel('Number of clusters', fontsize=tkfont)
                    # plt.ylabel('WCSS: Within-Cluster Sum of Square', fontsize=tkfont)
                    # plt.tick_params(labelsize=tkfont)
                    # fig.savefig('AQS_Ozone/'+sitename[s]+str(siteid[s])+'Elbow.png', dpi=300, bbox_inches='tight')
                    # plt.close()

                    # kmean cluster
                    if clusterm == 'kmeans':
                        kmeans = KMeans(n_clusters=clustern, init='k-means++', max_iter=300, random_state=0)#, n_init=10, random_state=0
                        if normm==1:
                            pred_y = kmeans.fit_predict(M_pn[M_datevalid[s, :] == 1, :])
                        else:
                            pred_y = kmeans.fit_predict(M_para[M_datevalid[s, :] == 1, :])
                        print(kmeans.inertia_)


                    elif clusterm == 'kmedoids':
                        kmedoids = KMedoids(n_clusters=clustern, method='pam', init='k-medoids++', max_iter=300, random_state=0)
                        if normm==1:
                            pred_y = kmedoids.fit_predict(M_pn[M_datevalid[s, :] == 1, :])
                        else:
                            pred_y = kmedoids.fit_predict(M_para[M_datevalid[s, :] == 1, :])

                    M_y[s, M_datevalid[s, :] == 1] = pred_y

                    # assigning groups
                    #
                    vaftnon = np.zeros(clustern)
                    vaftnon[:] = np.nan
                    vmornin = np.zeros(clustern)
                    vmornin[:] = np.nan
                    g_ind = np.zeros(clustern)
                    g_ind[:] = np.nan

                    #Cluname=['Sea Breeze','Oscillation','Southerly','Westerly']
                    #Cluname = ['1', '2', '3', '4']
                    Cluname = ['SB', 'O', 'S', 'W']

                    # for i in range(clustern):
                    #     g_ind[i]=i

                    for g in range(clustern):
                        Ug = M_U[np.where(M_y[s, :] == g), :]
                        Vg = M_V[np.where(M_y[s, :] == g), :]
                        Ul = np.nanmean(Ug, axis=1).reshape(24)
                        Vl = np.nanmean(Vg, axis=1).reshape(24)
                        C = (Ul * Ul + Vl * Vl)
                        # 4: 7])
                        # M_para[t, 2] = np.mean(M_U[t, 14:17])
                        vaftnon[g]=np.mean(Vl[14:17])
                        vmornin[g] =np.mean(Vl[4:7])
                    vaftnons = np.argsort(vaftnon)[::-1]  # desc
                    vmornins = np.argsort(vmornin)[::-1]
                    # w
                    g_ind[2]=vmornins[0]# large v in the marning - S
                    g_ind[1]=vmornins[3]# - v in the marning - O

                    if vaftnons[0] == vmornins[0]:
                        g_ind[0] = vaftnons[1]
                    else:
                        g_ind[0] = vaftnons[0]

                    for i in range(4):
                        if i not in g_ind:
                            g_ind[3] = i
                            break

                    for g in range(4):
                        Mgp[s, np.where(M_y[s, :] == g_ind[g])] = g  # assigned

                    with open(outpath + '/' + siteid[s] + clusterm + '_norm' + str(
                            normm) + '_' + '_' + str(para_n) + '_' + str(clustern) + 'CLusterAssign.txt', 'w') as testfile:
                        for row in Mgp[s, :]:
                            testfile.write(str(row) + '\n')

                    with open(outpath + '/'+ siteid[s] + clusterm + '_norm' + str(
                            normm) + '_' + '_' + str(para_n) + '_' + str(clustern) + 'CLusterOrig.txt', 'w') as testfile:
                        for row in M_y[s, :]:
                            testfile.write(str(row) + '\n')

                    with open(outpath + '/' + siteid[s] + clusterm + '_norm' + str(
                            normm) + '_' + '_' + str(para_n) + '_' + str(clustern) + 'CLusterDate.txt', 'w') as testfile:
                        for row in M_Date:
                            testfile.write(row.strftime('%Y-%m-%d') + '\n')

                    # # cluster average TS
                    cols = ['#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#a6cee3', '#1f78b4', '#b2df8a',
                            '#33a02c']  # light, dark, light, dark...




                    # tkfont=20
                    # ttfont = 30
                    for g in range(clustern):
                        fig = plt.figure(figsize=(12, 30))
                        Ug = Mt_U[np.where(M_y[s, :] == g_ind[g]), :]
                    
                    
                    
                        Vg = Mt_V[np.where(M_y[s, :] == g_ind[g]), :]
                        WSg = Mt_WS[np.where(M_y[s, :] == g_ind[g]), :]
                        Dateg = M_Date[np.where(M_y[s, :] == g_ind[g])]
                        Datetag = []
                        for i in range(Ug.shape[1]):
                            Datetag = np.append(Datetag, Dateg[i].strftime('%Y-%m-%d'))
                            Ul = Ug[0, i, :]
                            Vl = Vg[0, i, :]
                            WSl = WSg[0, i, :]

                            qq = plt.quiver(M_Hour, i, Ul, Vl, WSl, cmap=plt.cm.jet, scale=100)
                            plt.clim(0, 5)
                        plt.title(sitename[s] + ', ' + '' + Cluname[g], fontsize=ttfont, fontweight='bold')
                        # plt.ylabel('Date', fontsize=tkfont)
                        plt.xlabel('EST', fontsize=tkfont)
                        plt.yticks(range(Ug.shape[1]), Datetag)
                        # plt.legend(fontsize="xx-large")
                        # cbar=plt.colorbar(qq, cmap=plt.cm.jet)
                        # cbar.ax.tick_params(labelsize=tkfont)
                        plt.tick_params(labelsize=tkfont)
                        plt.ylim([-1, Ug.shape[1] + 1])
                        plt.xticks(np.arange(0, 24, 3))
                        fig.savefig(outpath+'/' +  siteid[s] + clusterm + '_norm'+str(normm) + '_'+'_'+str(para_n)+'_'+str(clustern)+'ClusterWind' + str(g) + '.png', dpi=300,
                                    bbox_inches='tight')
                        plt.close()

                    fig = plt.figure(figsize=(8, 8))
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
                        qq = plt.quiver(M_Hour, g, Ul, Vl, C, cmap=plt.cm.jet, scale=30)
                        plt.clim(0,5)
                        # str(Ug.shape[1])
                    #plt.title(sitename[s] + ', Mean Wind (' + clusterm + '_norm'+str(normm) + '_'+'_'+str(para_n)+')', fontsize=ttfont, fontweight='bold')
                    plt.title(
                        sitename[s] + ' Wind Clusters',
                        fontsize=ttfont, fontweight='bold')
                    plt.xlabel('EST', fontsize=tkfont)
                    # plt.legend(fontsize="xx-large")
                    cbar=plt.colorbar(qq, cmap=plt.cm.jet,fraction=0.04, pad=0.03)
                    cbar.ax.tick_params(labelsize=tkfont)
                    cbar.set_label('[m/s]', fontsize=20)

                    plt.tick_params(labelsize=tkfont)
                    plt.ylim([-0.5, clustern ])
                    plt.xticks(np.arange(0, 24, 3))
                    plt.yticks(np.linspace(0, clustern-1, num=clustern),Cluname[0:clustern])
                    fig.savefig(outpath+'/' + siteid[s]  +clusterm + '_norm'+str(normm) + '_'+'_'+str(para_n)+'_'+str(clustern)+'ClusterMean.png', dpi=300, bbox_inches='tight')
                    plt.close()


                    # # Plot morning and afternoon windroses
                    rosemax=100
                    fig = plt.figure(figsize=(25, 8))

                    ax2 = fig.add_subplot(1, 3, 1)
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
                        qq = ax2.quiver(M_Hour, 3-g, Ul, Vl, C, cmap=plt.cm.jet, scale=35,clim=[0,5])

                        # str(Ug.shape[1])
                    #plt.title(sitename[s] + ', Mean Wind (' + clusterm + '_norm'+str(normm) + '_'+'_'+str(para_n)+')', fontsize=ttfont, fontweight='bold')

                    ax2.set_xlabel('EST', fontsize=tkfont)
                    # plt.legend(fontsize="xx-large")
                    cbar=plt.colorbar(qq, cmap=plt.cm.jet,fraction=0.04)
                    cbar.ax.tick_params(labelsize=tkfont)
                    cbar.set_label('Wind Speed (m/s)', fontsize=20)

                    plt.tick_params(labelsize=tkfont)
                    plt.ylim([-0.5, clustern ])
                    plt.xticks(np.arange(0, 24, 4),
                               ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'])
                    plt.yticks(np.linspace(0, clustern-1, num=clustern),Cluname[::-1])#Cluname[0:clustern]

                    ax2.set_title('(a) Wind TS', fontsize=25, fontweight='bold',
                                      loc='left')

                    ax2.text(30, 4.1,'(b) Wind roses', fontsize=25, fontweight='bold')

                    ax2.text(72.5, 3, 'Morning', fontsize=22, fontweight='bold', horizontalalignment='left',
                             verticalalignment='center')
                    ax2.text(72.5
                             , 0.1, 'Afternoon', fontsize=22, fontweight='bold', horizontalalignment='left',
                             verticalalignment='center')
                    pos1 = [0.03, 0.1, 0.3, 0.8]
                    ax2.set_position(pos1)
                    cbar.ax.tick_params(labelsize=tkfont)

                    # fig.savefig(outpath+'/' + siteid[s]  +clusterm + '_norm'+str(normm) + '_'+'_'+str(para_n)+'_'+str(clustern)+'ClusterMean.png', dpi=300, bbox_inches='tight')
                    # plt.close()



                    bins_range=np.arange(0,6,1)
                    for g in range(clustern):
                        WSg = Mt_WS[np.where(M_y[s, :] == g_ind[g]), :]
                        Wdirg = Mt_WD[np.where(M_y[s, :] == g_ind[g]), :]

                        ax2 = fig.add_subplot(2, 6, g+1+2, projection='windrose')
                        ax2.bar(Wdirg[0, :, 4:7].reshape(WSg.shape[1]*3), WSg[0, :, 4:7].reshape(WSg.shape[1]*3), normed=True,
                                opening=0.8, edgecolor='white',bins=bins_range)
                        ax2.set_title( Cluname[g]+' '+str(WSg.shape[1])+'', fontsize=22, fontweight='bold')
                        print(WSg.shape[1])
                        ax2.set_yticks(np.arange(0, 40, step=10))
                        ax2.set_yticklabels(np.arange(0, 40, step=10), fontsize=20)
                        ax2.tick_params(labelsize=20)
                        pos1 = [0.39+0.13*g, 0.5, 0.13, 0.3]
                        ax2.set_position(pos1)

                        # ax2.grid(linewidth=0.5,
                        #              color='k', alpha=0.5, linestyle='--')

                        ax2 = fig.add_subplot(2,6,g+1+clustern+4,projection='windrose')
                        ax2.bar(Wdirg[0,:,14:17].reshape(WSg.shape[1]*3),WSg[0,:,14:17].reshape(WSg.shape[1]*3),normed=True,opening=0.8,edgecolor='white',bins=bins_range)
                        #ax2.set_title(Cluname[g]+' '+str(WSg.shape[1])+'', fontsize=20, fontweight='bold')
                        if g<2:
                            ax2.set_yticks(np.arange(0, 40, step=10))
                            ax2.set_yticklabels(np.arange(0, 40, step=10), fontsize=20)
                        else:
                            ax2.set_yticks(np.arange(0, 40, step=10))
                            ax2.set_yticklabels(np.arange(0, 40, step=10), fontsize=20)
                        ax2.tick_params(labelsize=20)
                        pos1 = [0.39+0.13*g, 0.1, 0.13, 0.3]
                        ax2.set_position(pos1)
                    l=ax2.legend(bbox_to_anchor=(1.1, 0.5), fontsize=20,frameon=False,title='Wind Speed\n(m/s)',title_fontsize=20)
                    plt.setp(l.get_title(), multialignment='center')

                    fig.savefig(outpath+'/' + siteid[s] +clusterm + '_norm'+str(normm) + '_'+'_'+str(para_n)+'_'+str(clustern)+ 'ClusterWindrose.png', dpi=600)
                    fig.savefig(outpath + '/' + siteid[s] + clusterm + '_norm' + str(normm) + '_' + '_' + str(
                        para_n) + '_' + str(clustern) + 'ClusterWindrose.pdf', dpi=600)
                    plt.close()

#slistcoast=['BRON','BKLN','MANH','QUEE','SOUT','STAT','STON','WANT']
# fn = 'WindClustertEST\TClusterQUEE_alldays\CLusterDateQueensQUEEkmeans_norm0__24_3.txt'
# with open(fn, 'r') as f:
#     text = f.read()
#     Metdate = text.split()
# slistcoast=['BKLN','QUEE','STAT','WANT']
#
# i=0
# for sid in range(len(siteid)):
#     if siteid[sid] in slistcoast:
#         i=i+1
#         if i==1:
#             sinind=sid
#         else:
#             sinind = np.append(sinind, sid)
#
# fig, ax = plt.subplots(1,1)
# plt.imshow(Mgp[sinind,:], aspect='auto', interpolation='none',
#             origin='lower')
# ax.set_yticks([0,1,2,3])
# ax.set_yticklabels(slistcoast)
# plt.title('kmeans')
# plt.savefig(outpath+'/Coastal.png', dpi=300, bbox_inches='tight')
# plt.close()
#
# for i in range(4):
#     print(sum(sum(Mgp[sinind,:]-Mgp[sinind[i],:]==0)))
