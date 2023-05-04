#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tuesday April 06 2021
Dr. Mickaël Tharaud
Université Paris Cité / Institut de Physique du Globe de Paris
"""
############ LIBRAIRIES ############################################################################################################
import os, glob

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

############ USER INPUTS ############################################################################################################

### Please enter the parameters in "user_inputs" in the following order in form of a tuple() where each element is separated by a comma :
# one file extension [!and exception if applicable], one column separator, one keyword to stop the row skiping for the header, one keyword for column choice, one colormap for nice color

### Note that the file extension must be one of the following :
# .csv OR . txt

### Note that information about the file(s) to use can be inserted into the brackets
# ex: *[!_raw].csv --> all csv except the one(s) with '_raw' in the name

### Note that the column separator must be one of the following :
# ',', ';', '/'

### Note that the keyword to stop the row skiping for the header must be the name in the first raw of the first column of the dataframe in the file
# ex: 'event number'

### Note that keyword for column choice must be in at least one of the column header
# ex: 'mole %' or 'mass %'

### Note that the colormap must be one of the following :
# 'viridis', 'plasma', 'inferno', 'magma', 'cividis'

#==================================================================================================== WORKING DIRECTORIES
#====================================================================================================
#====================================================================================================

IWD = os.getcwd()

#==================================================================================================== CLASSES & FUNCTIONS
#====================================================================================================
#====================================================================================================

class myDataframe:
    ## Put all the data in a dataframe
    def __init__(self, entire_path, sep, keyword_1):
        self.path = entire_path
        self.sep = sep #= ','
        self.kw1 = keyword_1 #='event number'

    ## skip hearders based on the "event number" row position
    def skip_to(self): # line and sep to adjust to the file
        if os.stat(self.path).st_size == 0:
            raise ValueError("File is empty")
        with open(self.path) as f:
            pos = 0
            cur_line = f.readline()
            while not cur_line.startswith(self.kw1):
                pos = f.tell()
                cur_line = f.readline()
            f.seek(pos)
            self.df = pd.read_csv(f, sep=self.sep)

            return self.df


class myDf4Clust:
    ## create a subset dataframe for the clustering
    def __init__(self, dataframe, keyword_2):
        self.df = dataframe
        self.kw2 = keyword_2

    ## list of column with the specific keyword
    def column_name(self):
        col_name = list(self.df.columns.values)
        #print('Name(s) of the column(s):')
        #print(col_name,'\n')
        self.selected_col = [name for name in col_name if self.kw2 in name]

        return self.selected_col

    ## create a subset dataframe with selected column
    def subset_df(self):
        # new dataframe with selected column
        self.sub_df = self.df[self.column_name()]
        # sort column by alphabetic order
        self.sub_df = self.sub_df.reindex(sorted(self.sub_df.columns), axis=1)

        return self.selected_col, self.sub_df


class myBestThreshold:
    ## determine the optimal threshold for stopping the agglomerative clustering
    def __init__(self, dataframe, max_distance):
        self.df = dataframe
        self.max_distance = max_distance
        self.list_threshold = []
        self.list_score = []

    def optimal_threshold(self):
        self.best_score = -1
        self.best_threshold = None
        nb_labels = 2

        for self.threshold in np.linspace(1, self.max_distance, 100):
            self.model = AgglomerativeClustering(n_clusters=None, distance_threshold=self.threshold, compute_distances=True)
            self.labels = self.model.fit_predict(self.df.to_numpy())
            nb_labels = len(np.unique(self.labels))

            if nb_labels < 2:
                break

            self.score = silhouette_score(self.df.to_numpy(), self.labels, metric='euclidean')

            if self.score > self.best_score:
                self.best_score = self.score
                self.best_threshold = self.threshold

            self.list_threshold.append(self.threshold)
            self.list_score.append(self.score)

        return self.list_threshold, self.list_score, float(self.best_threshold)

        '''print('\n * The optimal distance threshold is: {:.1f}'.format(self.best_threshold) + '\n * and its cubic root is: {:.1f}'.format(self.best_threshold**(1/3)))

        # Plot the corresponding graph
        plt.title('Silhouette score vs Distance threshold', fontsize=12, fontstyle='italic')
        plt.scatter(self.list_threshold, self.list_score, c='k', marker='+')
        plt.axvline(x=self.best_threshold, c='r', ls=':')
        plt.annotate('Optimal threshold = {:.1f}'.format(self.best_threshold), xy = ((self.best_threshold)*1.02, np.mean(self.list_score)), fontsize = 'x-small', color = 'red')
        plt.ylabel('Silhouette score')
        plt.xlabel('Distance threshold')
        plt.show()'''

    def set_threshold(self):
        # set the distance threshold !!
        ## It must be an integer
        ### ex: 1, 2, 3...
        ## Or a float
        ### ex: 1.2, 2.5, 3.7...
        #### WARNING!! on the y-axis, the cubic root of the distance is represented
        #### if you want to set the threshold at 1.5 (read on y-axis), enter 3.375 --> 1.5**3
        #### otherwise if you enter 3, the threshold will be set at 1.4422
        self.set_thre = input('\n * Please set the distance threshold\n * Cubic root of:')

        return float(self.set_thre)


class myAgglomClust:
    ## perform the clustering
    def __init__(self, dataframe, distance_threshold):
        self.df = dataframe
        self.threshold = distance_threshold

    def clustering(self):
        self.clust = AgglomerativeClustering(n_clusters=None, distance_threshold= self.threshold, compute_distances=True)
        self.model = self.clust.fit(self.df.to_numpy())
        self.labels = self.clust.labels_

        return self.model, self.labels


class myLabels_in_df:
    ## Insert the labels in the dataframe
    def __init__(self, dataframe, labels):
        self.df = dataframe
        self.labels = labels

    def merging(self):
        # insert cluster number into the dataframe
        self.df['Cluster'] = self.labels
        # sort dataframe by cluster number
        self.df.sort_values(by=['Cluster'], inplace=True)

        return self.df


class myLists:
    ## Create lists to be used later
    def __init__(self, dataframe, selected_col):
        self.df = dataframe
        self.selected_col = selected_col
        self.particle_nb = 0
        self.list_cluster = []
        self.list_cluster_count = []
        self.list_cluster_prop = []

    def lists(self):
        # number of rows (particles) in the dataframe
        self.particle_nb = len(self.df)
        # list of the cluster number
        self.list_cluster = self.df['Cluster'].unique().tolist()

        # list of the cluster proportion
        for i in self.list_cluster:
            count = np.count_nonzero(self.df['Cluster'] == i)
            self.list_cluster_count.append(count)
            percent = count/(len(self.df['Cluster'].index))*100
            self.list_cluster_prop.append(percent)

        return self.particle_nb, self.list_cluster, self.list_cluster_count, self.list_cluster_prop

class myLinkageMatrix:
    ## Compute the linkage matrix and then plot the initial dendrogram
    def __init__(self, model):
        self.model = model
        self.threshold = []
        self.linkage_matrix = []
        self.wd = []

    def link_mat(self):
        ## linkage matrix
        # create the counts of samples under each node
        counts = np.zeros(self.model.children_.shape[0])
        n_samples = len(self.model.labels_)
        for i, merge in enumerate(self.model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        self.linkage_matrix = np.column_stack([self.model.children_, (self.model.distances_)**(1/3), counts]).astype(float)

        return self.linkage_matrix


class myDendrogram:
    ## Compute the linkage matrix and then plot the dendrogram
    def __init__(self, model, distance_threshold, link_mat, list_cluster, labels, cmap):
        self.model = model
        self.threshold = distance_threshold
        self.list_cluster = list_cluster
        self.labels = labels
        self.linkage_matrix = link_mat
        self.list_colors = []
        self.cmap = cmap

    def draw_tree(self):
        # list of colors for clusters
        cmap_cluster = plt.get_cmap(self.cmap, np.max(self.list_cluster)+1)
        cmap_colors = cmap_cluster.colors
        for i in range(len(cmap_colors)):
            rgba = cmap_colors[i]
            # rgb2hex accepts rgb or rgba
            name = rgb2hex(rgba)
            self.list_colors.append(name)

        #set the colors for the clusters
        set_link_color_palette(self.list_colors)

        # Plot the corresponding dendrogram
        plt.title('Hierarchical Clustering Dendrogram', fontsize=12, fontstyle='italic')
        dn = dendrogram(self.linkage_matrix, truncate_mode="level", p=10, color_threshold= self.threshold**(1/3), above_threshold_color='k', leaf_rotation=90, labels = self.labels, no_labels=True)
        plt.axhline(y=self.threshold**(1/3), c='r', ls=':')
        plt.annotate('Distance threshold = {}'.format(self.threshold), xy = (0.5,(self.threshold**(1/3))*1.02), fontsize = 'x-small', color = 'red')
        plt.ylabel('$\sqrt[3]{distance}$')
        plt.xlabel('Datapoints')
        #plt.show()

        ## some code for the consistency of cluster colors --> sort the colors based on the cluster number position
        cluster_pos = []
        for l, m in enumerate(self.list_cluster):
            #print(m)
            position = dn['ivl'].index(m)
            cluster_pos.append(position)

        # set the color order for the following figures
        cluster_order = [x for _,x in sorted(zip(cluster_pos,self.list_cluster))]
        self.list_colors = [x for _,x in sorted(zip(cluster_order,self.list_colors))]

        return self.list_colors


class myPie:
    def __init__(self, nb_part, list_cluster, list_cluster_prop, list_colors):
        self.nb_part = nb_part
        self.list_cluster = list_cluster
        self.list_cluster_prop = list_cluster_prop
        self.list_colors = list_colors

    ## draw the pie chart
    def draw_pie(self):
        # First Ring (outside)
        fig, ax = plt.subplots()

        # pie sizes
        radius = 1
        size = 0.4

        # Draw pie
        self.mypie, self.text = ax.pie(self.list_cluster_prop, radius=radius, labels=self.list_cluster, colors= self.list_colors, labeldistance=0.8, normalize=True)
        plt.setp(self.mypie, width=size, edgecolor='white')
        plt.setp(self.text, color= 'white', fontweight='bold')

        # Draw arrows and boxes for %
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="w", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")

        if len(self.list_cluster) == 1: # in order to draw the pie even if the cluster prop is 100%
            ang = 90
            y = 0.9
            x = 0
            horizontalalignment = 'center'
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate('{:.1f}%'.format(100), xy=(x, y), xytext=(0, 1.4*y), horizontalalignment=horizontalalignment, **kw)

        else:
            for i, p in enumerate(self.mypie):
                ang = (p.theta2 - p.theta1)/2. + p.theta1
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                connectionstyle = "angle,angleA=0,angleB={}".format(ang)
                kw["arrowprops"].update({"connectionstyle": connectionstyle})
                ax.annotate('{:.1f}%'.format(self.list_cluster_prop[i]), xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y), horizontalalignment=horizontalalignment, **kw)

        ax.set_aspect("equal")
        ax.set_title('Cluster\nproportions' '\n('+str(self.nb_part)+' part.)', x=0.5, y=0.45, fontsize=12, fontstyle='italic')

        plt.tight_layout()
        #plt.show()

        return self.mypie


class myHorizBar:
    def __init__(self,dataframe, selected_col, list_cluster, list_cluster_count, list_colors):
        self.df = dataframe
        self.selected_col = selected_col
        self.list_cluster = list_cluster
        self.list_cluster_count = list_cluster_count
        self.list_colors = list_colors

    ## plot one barh plot per cluster
    def draw_hbar(self):

        # ensure good representation/distribution of graphs
        self.mybarh, ax = plt.subplots(1, len(self.list_cluster), sharex=True, sharey=True)

        # extract isotope from the selected columns
        for char in self.selected_col:
            self.isotopes = [char.split(' ', maxsplit=1)[0] for char in self.selected_col]
            self.x_label = char.split(' ', maxsplit=1)[1]

        #ax = ax.ravel()
        print('\n * The average composition per cluster is:')

        for k in self.list_cluster:
            print('Cluster' + str(k) + ' -->')
            # create lists for mean and sd of cluster compo
            list_mean = []
            list_sd = []
            for l, m in enumerate(self.selected_col):
                mean = [np.mean(self.df[m][self.df['Cluster'] == k])]
                sd = [np.std(self.df[m][self.df['Cluster'] == k])]
                list_mean.append(mean)
                list_sd.append(sd)

                print('- ' + str(m) +': {:.1f} %'.format(mean[-1]*100)+' +/- {:.1f} %'.format(sd[-1]*100))

            list_mean = list(np.concatenate(list_mean).flat)
            list_sd = list(np.concatenate(list_sd).flat)

            if len(self.list_cluster) > 1:
                # plot barh
                ax[k].barh(np.arange(len(list_mean)), list_mean, xerr=list_sd, align='center', color=self.list_colors[k])
                ax[k].set_yticks(np.arange(len(self.isotopes)))
                ax[k].set_yticklabels(self.isotopes)
                ax[k].invert_yaxis()  # labels read top-to-bottom
                ax[k].set_xlabel(self.x_label)
                ax[k].set_title('Cluster n°'+str(k)+'\n('+str(self.list_cluster_count[k])+' part.)', fontsize=10)

            else:
                ax.barh(np.arange(len(list_mean)), list_mean, xerr=list_sd, align='center', color=self.list_colors[k])
                ax.set_yticks(np.arange(len(self.isotopes)))
                ax.set_yticklabels(self.isotopes)
                ax.invert_yaxis()  # labels read top-to-bottom
                ax.set_xlabel(self.x_label)
                ax.set_title('Cluster n°'+str(k)+'\n('+str(self.list_cluster_count[k])+' part.)', fontsize=10)

            print('\n')
            # clear lists
            list_mean.clear()
            list_sd.clear()

        #plt.suptitle(self.kw2+' composition of:', fontsize=12, fontstyle='italic')
        #plt.tight_layout()
        #plt.show()

        return self.mybarh


class myHist:
    def __init__(self, dataframe, list_cluster, list_cluster_count, list_colors):
        self.df = dataframe
        self.list_cluster = list_cluster
        self.list_cluster_count = list_cluster_count
        self.list_colors = list_colors

    ## plot multiple histogram
    def draw_hist(self):
        # ensure good representation/distribution of graphs
        self.myhist, ax = plt.subplots(1, len(self.list_cluster), sharex=True, sharey=True) #figsize=(15, 15)

        #ax = ax.ravel()

        # determine histogram limits
        h_min = np.min(self.df['total mass [fg]'])
        h_max = np.max(self.df['total mass [fg]'])
        h_range = (h_min, h_max)
        h_bins = int(np.sqrt(self.df['total mass [fg]'].shape[0]))

        print('* The average total mass per cluster is:')

        # plot the histogram
        for k in self.list_cluster:
            if len(self.list_cluster) > 1:
                ax[k].hist(self.df['total mass [fg]'][self.df['Cluster'] == k], bins=h_bins, range=h_range, density=True, color=self.list_colors[k])
                ax[k].set_title('Cluster n°'+str(k)+'\n('+str(self.list_cluster_count[k])+' part.)', fontsize=10)
                mean = np.mean(self.df['total mass [fg]'][self.df['Cluster'] == k])
                sd = np.std(self.df['total mass [fg]'][self.df['Cluster'] == k])
                ax[k].annotate('$\overline{m}$ '+'= {:.1f} fg'.format(mean)+'\n$\pm$ '+'{:.1f} fg'.format(sd), xy=(0.05, 0.9), xycoords='axes fraction', fontweight='bold')
                ax[k].set_xlabel('total mass [fg]')
                ax[0].set_ylabel('Density')

            else:
                ax.hist(self.df['total mass [fg]'][self.df['Cluster'] == k], bins=h_bins, range=h_range, density=True, color=self.list_colors[k])
                ax.set_title('Cluster n°'+str(k)+'\n('+str(self.list_cluster_count[k])+' part.)', fontsize=10)
                mean = np.mean(self.df['total mass [fg]'][self.df['Cluster'] == k])
                sd = np.std(self.df['total mass [fg]'][self.df['Cluster'] == k])
                ax.annotate('$\overline{m}$ '+'= {:.1f} fg'.format(mean)+'\n$\pm$ '+'{:.1f} fg'.format(sd), xy=(0.5, 0.9), xycoords='axes fraction', fontweight='bold')
                ax.set_xlabel('total mass [fg]')
                ax.set_ylabel('Density')

            print('Cluster' + str(k) + ' --> {:.3f} fg'.format(mean)+' +/- {:.3f} fg'.format(sd))

        #print(self.df)
        print('\n')
        print('* The total mass of NPs in the sample is ' + '{:.1f} fg'.format(np.sum(self.df['total mass [fg]'])))
        #plt.suptitle('Particle mass [fg] distribution of:', fontsize=12, fontstyle='italic')
        #plt.tight_layout()
        #plt.show()

        return self.myhist


class mySize:
    def __init__(self,dataframe, selected_col):
        self.df = dataframe
        self.selected_col = selected_col

    def size_calculation(self):
        self.df_density = pd.read_csv(IWD+'/Density.csv', sep=';')  #pd.DataFrame(data = DENSITY)

        for i, j in enumerate(self.selected_col):
            self.df[j[0:-2]+' [fg]'] = self.df[j]*self.df['total mass [fg]']
            self.df[j[0:-7]+' volume [nm3]'] = self.df[j[0:-2]+' [fg]']/(self.df_density['Density'][self.df_density['Isotope']==j[0:-7]].iloc[0]*(10**(-6)))
            self.df[j[0:-7]+' diameter [nm]'] = (6*self.df[j[0:-7]+' volume [nm3]']/np.pi)**(1/3)

        self.df['total volume [nm3]'] = self.df[[col for col in self.df.columns if col.endswith('volume [nm3]')]].sum(axis=1)
        self.df['total diameter [nm]'] =  (6*self.df['total volume [nm3]']/np.pi)**(1/3)

        return self.df


class myBoxPlot:
    def __init__(self, dataframe,list_cluster, list_colors):
        self.df = dataframe
        self.list_cluster = list_cluster
        self.list_colors = list_colors

    def draw_boxplot(self):
        fig, ax = plt.subplots()

        #ax = ax.ravel()

        print('\n * The average eq. spherical diameter per cluster is:')

        for k in self.list_cluster:
            self.myboxplot = ax.boxplot(self.df['total diameter [nm]'][self.df['Cluster']==k], positions = [k], patch_artist=True)
            plt.setp(self.myboxplot["boxes"], facecolor=self.list_colors[k])

            print('Cluster' + str(k) + ' --> {:.1f} nm'.format(np.median(self.df['total diameter [nm]'][self.df['Cluster']==k])) + ' +/- {:.1f} nm'.format(np.std(self.df['total diameter [nm]'][self.df['Cluster']==k])))

        plt.xlabel('Cluster n°', fontsize=10)
        plt.ylabel('Eq. spherical diameter [nm]', fontsize=10)
        plt.suptitle('Eq. spherical diameter [nm] distribution of:', fontsize=12, fontstyle='italic')
        plt.tight_layout()
        #plt.show()

        return self.myboxplot


class myResults:
    def __init__(self, entire_path, dataframe, epsilon):
        self.path = entire_path
        self.df = dataframe
        self.epsilon = epsilon

    def save_df_to_csv(self):
        # prepare the df to save the threshold directly into the csv
        #self.df.rename(columns={'Cluster': 'Cluster d'+str(self.epsilon)}, inplace=True)

        # save the results in the "Results" folder. Create it if it does not exist
        shorten_path, filename = os.path.split(self.path)
        result_directory = 'Results'
        if not os.path.exists(shorten_path + '/' + result_directory):
            os.makedirs(shorten_path + '/' + result_directory)

        fullname = os.path.join(shorten_path, result_directory, filename[:-4]+'_results.TXT')
        self.df.to_csv(fullname)



#==================================================================================================== RUN THE CODE
#====================================================================================================
#====================================================================================================

def calculate_threshold(path, separator = ',', skip_stop = 'event number', column_kw = 'mole %'):
    mydf = myDataframe(path, separator, skip_stop)
    my_original_df = mydf.skip_to()

    mydf2 = myDf4Clust(my_original_df, column_kw)
    selected_columns, mydf4clust = mydf2.subset_df()

    mythreshold = myBestThreshold(mydf4clust, 20)
    mythresholds, myscores, myopt_thresold = mythreshold.optimal_threshold()

    return my_original_df, selected_columns, mydf4clust, mythresholds, myscores, myopt_thresold


def clustering(path, my_original_df, selected_columns, mydf4clust, distance_threshold, colormap = 'viridis'):

    myclust = myAgglomClust(mydf4clust, distance_threshold)
    model, labels = myclust.clustering()

    mylabels = myLabels_in_df(my_original_df, labels)
    my_intermediate_df = mylabels.merging() # put the labels into the original df

    mylists = myLists(my_intermediate_df, selected_columns)
    Particle_nb, Cluster_nb, Cluster_part_nb, Cluster_prop = mylists.lists()

    mylinkmat = myLinkageMatrix(model)
    linkmat = mylinkmat.link_mat()

    mydendro = myDendrogram(model, distance_threshold, linkmat, Cluster_nb, labels, colormap)
    mycolors = mydendro.draw_tree()

    mypie = myPie(Particle_nb, Cluster_nb, Cluster_prop, mycolors)
    mypie.draw_pie()

    mysize = mySize(my_intermediate_df, selected_columns)
    my_final_df = mysize.size_calculation()

    myboxplot = myBoxPlot(my_final_df, Cluster_nb, mycolors)
    myboxplot.draw_boxplot()

    #plt.show() # shows all the plots

    myresults = myResults(path, my_final_df, distance_threshold)
    myresults.save_df_to_csv()

    return my_final_df, Cluster_nb, Cluster_part_nb, mycolors

'''if __name__ == '__main__':
    clustering()'''
