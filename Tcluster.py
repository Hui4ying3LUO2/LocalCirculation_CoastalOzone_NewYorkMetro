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
outpath='3TSolarCluster/'
clustern = 3
Cluname=['Hot','Moderate','Cool']#Cluname=['Clear','LightClouds','PMClouds','Cloudy','Overcast']
datam='single' # region
# cluster_M=['kmeans','kmeans','kmeans','kmeans','kmedoids','kmedoids','kmedoids','kmedoids']
# Norm_YN=[1,1,0,0,1,1,0,0]
# para_N=[4,3,4,3,4,3,4,3]
cluster_M=['kmeans','kmedoids','kmeans','kmedoids']
Norm_YN=[0,0,1,1]
para_N=[24,24,24,24]#[7,7,7,7]#[5,5,5,5]#[6,6,6,6]#[3,3,3,3]#[4,4,4,4]#
#metlist=['SDHN4','ROBN4','QPTR1','MHRN6','KPTN6','BRHC3','44069','44065','44040','44039','WANT','STON','STAT','SOUT','QUEE','MANH','BRON','BKLN','FOK','HWV','NYC','MTP','FRG','TEB','CDW','BDR']
metlist=['QUEE']

yearlist=['2017','2018','2019']
days_n = 92*len(yearlist)

ttfont = 20
tkfont = 20
#############
def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())

######### Input loading

Met_T = []# use asos 2015-2019
#

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


M_Time = np.array(Met_T).reshape(days_n, 24)
M_Hour = np.arange(24)
M_Date = M_Time[:, 0]
M_datevalid = np.zeros([len(siteid), days_n])
M_datevalid[:] = 1

#metlist=np.concatenate((metlist,site_id_aqs))

for mid in [0]:#(len(para_N)):

    clusterm = cluster_M[mid]  # kmedoids'# kmeans
    normm = Norm_YN[mid]  #
    para_n =  para_N[mid]#


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

            solar_sitep = pd.DataFrame(data=solar_site)
            solar_sitefit = solar_sitep.interpolate()
            M_solar = solar_sitefit.to_numpy().reshape(days_n, 24)

            # mark days not satisfy 90% data coverage, > 2days; set data to 0 for those days
            for t in range(len(M_Date)):
                if sum(np.isnan(Mt_TEMP[t, :])) > 2 :# 11 12 13 EST
                    M_datevalid[s, t] = 0
                    M_V[t, :] = 0
                    M_U[t, :] = 0
                    M_WD[t, :] = 0
                    M_WS[t, :] = 0
                    M_PS[t, :] = 0
                    M_TEMP[t, :] = 0
                    M_solar[t, :] = 0


            if sum(M_datevalid[s, :]) > 30:  # more than 30 valid days sitem_id[s]=='BKLN': #s in range(len(sitem_id)):##

                # features calculation and normalization
                # A: original, normalized 7
                if para_n == 24:
                    for t in range(len(M_Date)):
                        if M_datevalid[s, t] == 1:
                            M_para[t, :] = M_TEMP[t, :]  # 3:7

                # kmean cluster
                if clusterm == 'kmeans':
                    kmeans = KMeans(n_clusters=clustern, init='k-means++', max_iter=300, n_init=10, random_state=0)
                    if normm==1:
                        pred_y = kmeans.fit_predict(M_pn[M_datevalid[s, :] == 1, :])
                    else:
                        pred_y = kmeans.fit_predict(M_para[M_datevalid[s, :] == 1, :])


                elif clusterm == 'kmedoids':
                    kmedoids = KMedoids(n_clusters=clustern, method='pam', init='k-medoids++', max_iter=300, random_state=0)
                    if normm==1:
                        pred_y = kmedoids.fit_predict(M_pn[M_datevalid[s, :] == 1, :])
                    else:
                        pred_y = kmedoids.fit_predict(M_para[M_datevalid[s, :] == 1, :])

                M_y[s, M_datevalid[s, :] == 1] = pred_y

                # assigning groups
                vaftnon = np.zeros(clustern)
                vaftnon[:] = np.nan
                vmornin = np.zeros(clustern)
                vmornin[:] = np.nan
                g_ind = np.zeros(clustern)
                g_ind[:] = np.nan



                for i in range(clustern):
                    g_ind[i]=i


                for g in range(clustern):
                    solarg = M_solar[np.where(M_y[s, :] == g), :]
                    solarl = np.nanmean(solarg, axis=1).reshape(24)
                    vaftnon[g]=np.mean(solarl[14:17])

                noons = np.argsort(vaftnon)[::-1]  # desc
                # w
                for g in range(clustern):
                    g_ind[g]=noons[g]# large v in the marning - S

                for g in range(clustern):
                    Mgp[s, np.where(M_y[s, :] == g_ind[g])] = g  # assigned

                # cluster average TS
                cols = ['#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#a6cee3', '#1f78b4', '#b2df8a',
                        '#33a02c','#e9a3c9','#c51b7d','#998ec3','#542788',]  # light, dark, light, dark...

                for g in range(clustern):
                    fig = plt.figure(figsize=(8, 8))

                    solarg = Mt_TEMP[np.where(M_y[s, :] == g_ind[g]), :]
                    for i in range(solarg.shape[1]):
                        solarl = solarg[0, i, :]
                        qq = plt.plot(M_Hour, solarl, color=cols[2*g])
                    C = np.nanmean(solarg, axis=1).reshape(24)
                    qq = plt.plot(M_Hour, C,color=cols[2*g+1])

                    plt.title(sitename[s] + ', '+ Cluname[g]+' (' + clusterm + '_norm'+str(normm) + '_'+'_'+str(para_n)+')', fontsize=ttfont, fontweight='bold')
                    plt.xlabel('EST', fontsize=tkfont)
                    # plt.legend(fontsize="xx-large")
                    plt.tick_params(labelsize=tkfont)
                    plt.ylim([8, 38])
                    plt.xticks(np.arange(0, 24, 3))
                    fig.savefig(outpath+'/T' + sitename[s] + siteid[s]  +clusterm + '_norm'+str(normm) + '_'+'_'+str(para_n)+'_'+str(clustern)+'ClusterMean'+str(g)+'.png', dpi=300, bbox_inches='tight')
                    plt.close()

                fig = plt.figure(figsize=(8, 8))
                for g in range(clustern):

                    O3g =  Mt_TEMP[np.where(M_y[s, :] == g_ind[g]), :]
                    # Ozone
                    mett=np.nanpercentile(O3g, 50, axis=1).reshape(24)


                    plt.plot(mett, color=cols[g * 2 + 1], label= Cluname[g]+' ' + str(len(np.where(M_y[s, :] == g_ind[g])[0])))
                    plt.plot([np.min(np.where(mett==np.max(mett)))],np.max(mett),'o', color=cols[g * 2 + 1])
                    plt.plot([np.min(np.where(mett == np.min(mett)))], np.min(mett), 'o', color=cols[g * 2 + 1])
                    print(np.max(mett))
                    print(np.min(mett))
                    plt.fill_between(M_Hour, np.nanpercentile(O3g, 25, axis=1).reshape(24),
                                      np.nanpercentile(O3g, 75, axis=1).reshape(24),
                                      alpha=0.3, facecolor=cols[g * 2])

                    #plt.title(sitename[s] + ', Med T'+ ' (' + clusterm + '_norm'+str(normm) + '_'+'_'+str(para_n)+')', fontsize=ttfont, fontweight='bold')
                    plt.title(sitename[s] + ' T Clusters', fontsize=ttfont, fontweight='bold')
                    plt.xlabel('EST', fontsize=tkfont)
                    plt.ylabel('T [C]', fontsize=tkfont)
                    plt.legend(frameon=False, fontsize=20,loc='lower center')
                    plt.tick_params(labelsize=tkfont)
                    plt.ylim([8, 38])
                    plt.xticks(np.arange(0, 24, 3))
                fig.savefig(outpath+'/T' + sitename[s] + siteid[s]  +'ClusterMed.png', dpi=300, bbox_inches='tight')
                plt.close()

            with open(outpath+'/TCLusterAssign' + siteid[s]  +clusterm + '_norm'+str(normm) + '_'+'_'+str(para_n)+'_'+str(clustern)+'.txt', 'w') as testfile:
                for row in Mgp[s,:]:
                    testfile.write(str(row) + '\n')

            with open(outpath + '/TCLusterOrig' +  siteid[s]  +clusterm + '_norm'+str(normm) + '_'+'_'+str(para_n)+'_'+str(clustern)+'.txt', 'w') as testfile:
                for row in M_y[s,:]:
                    testfile.write(str(row) + '\n')

            with open(outpath + '/TCLusterDate' + siteid[s]  +clusterm + '_norm'+str(normm) + '_'+'_'+str(para_n)+'_'+str(clustern)+'.txt', 'w') as testfile:
                for row in M_Date:
                    testfile.write(row.strftime('%Y-%m-%d') + '\n')