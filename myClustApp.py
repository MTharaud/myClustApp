import os
from PyQt5.QtWidgets import QApplication, QFileDialog, QListWidget, QMainWindow, QWidget, QHBoxLayout, QGridLayout,QRadioButton, QPushButton, QLabel, QTextEdit, QLineEdit, QSizePolicy
from PyQt5.QtGui import QDoubleValidator
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import myHAC
import myDBScan

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("spICP-ToF-MS data clustering")
        self.setGeometry(200, 100, 1600, 1000)

        # Create a left widget to hold the layout on the left
        widget = QWidget()
        self.setCentralWidget(widget)

        main_grid = QHBoxLayout()
        widget.setLayout(main_grid)

        left_grid = QGridLayout()
        main_grid.addLayout(left_grid, 1)
        right_grid = QGridLayout()
        main_grid.addLayout(right_grid, 10)

        # Create a button to browse for a directory
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_for_directory)
        left_grid.addWidget(browse_button, 0, 0, 1, 2)

        # Create a list widget to display the CSV files
        self.csv_list_widget = QListWidget()
        #self.csv_list_widget_title = QLabel('Double-click to select')
        self.csv_list_widget.setFixedHeight(125)
        self.csv_list_widget.itemDoubleClicked.connect(self.file_selected)
        self.csv_list_widget.itemDoubleClicked.connect(self.enable_button)
        #grid.addWidget(self.csv_list_widget_title, 0, 1, 1, 1)
        left_grid.addWidget(self.csv_list_widget, 1, 0, 1, 2)

        # Create two radio buttons
        self.qrb_title = QLabel('Select methodology:')
        self.qrb1 = QRadioButton("Agglomerative clustering")
        self.qrb2 = QRadioButton("Density Based clustering")
        self.qrb1.setChecked(True)
        left_grid.addWidget(self.qrb_title, 2, 0, 1, 1)
        left_grid.addWidget(self.qrb1, 2, 1, 1, 1)
        left_grid.addWidget(self.qrb2, 3, 1, 1, 1)

        # Create a button to calculate the optimal threshold or epsilon
        self.start_button = QPushButton("Calculate optimal value")
        self.start_button.setDisabled(True) # Button only active if one item is selected (double clicked)
        self.start_button.clicked.connect(self.calculate_opt_value)
        self.start_button.clicked.connect(self.enable_button_2)
        self.start_button.clicked.connect(self.draw_silhouette)
        left_grid.addWidget(self.start_button, 4, 0, 1, 2)

        # Create an empty text edit to display the optimal threshold
        self.recommended_label = QLabel('Recommended Threshold (HAC) or Epsilon (DBScan):')
        self.recommended_text = QTextEdit()
        self.recommended_text.setReadOnly(True)
        self.recommended_text.setFixedHeight(25)
        left_grid.addWidget(self.recommended_label, 5, 0, 1, 1)
        left_grid.addWidget(self.recommended_text, 5, 1, 1, 1)

        # Create the matplotlib figure and canvas
        self.fig_silhouette = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig_silhouette)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Create the matplotlib toolbar and add it to the window
        self.toolbar = NavigationToolbar(self.canvas, self)
        left_grid.addWidget(self.toolbar, 6, 0, 1, 2)
        left_grid.addWidget(self.canvas, 7, 0, 1, 2)

        # Create an empty text edit to set the optimal threshold or epsilon
        self.set_label = QLabel('Set desired Threshold (HAC) or Epsilon (DBScan):')
        self.set_text = QLineEdit()
        self.set_text.setValidator(QDoubleValidator(0.00,99.999,3))
        left_grid.addWidget(self.set_label, 8, 0, 1, 1)
        left_grid.addWidget(self.set_text, 8, 1, 1, 1)

        # Create a button to start the clustering
        self.start_button_2 = QPushButton("Perform clustering")
        self.start_button_2.setDisabled(True) # Button only active if one item is selected (double clicked)
        self.start_button_2.clicked.connect(self.start_clustering)
        self.start_button_2.clicked.connect(self.draw_hbar)
        self.start_button_2.clicked.connect(self.draw_hist)
        self.start_button_2.setStyleSheet("background-color: lightblue")
        left_grid.addWidget(self.start_button_2, 9, 0, 1, 2)

        self.fig_horizbar = Figure()
        self.canvas2 = FigureCanvas(self.fig_horizbar)
        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Create the matplotlib toolbar and add it to the window
        self.toolbar2 = NavigationToolbar(self.canvas2, self)
        right_grid.addWidget(self.toolbar2, 0, 0, 1, 1)
        right_grid.addWidget(self.canvas2, 1, 0, 1, 10)

        self.fig_mass_distrib = Figure()
        self.canvas3 = FigureCanvas(self.fig_mass_distrib)
        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Create the matplotlib toolbar and add it to the window
        self.toolbar3 = NavigationToolbar(self.canvas3, self)
        right_grid.addWidget(self.toolbar3, 2, 0, 1, 1)
        right_grid.addWidget(self.canvas3, 3, 0, 1, 10)

    def browse_for_directory(self):
        # Open a file dialog to select a directory
        self.directory = QFileDialog.getExistingDirectory(self, "Select Directory")

        # If a directory was selected, display the CSV files in the list widget
        if self.directory:
            csv_files = [filename for filename in os.listdir(self.directory) if filename.endswith(".csv")]
            self.csv_list_widget.clear()
            self.csv_list_widget.addItems(csv_files)

    def file_selected(self, item):
        # Get the selected file from the list widget
        self.selected_file = item.text()
        print(f"\n * Selected file: {self.selected_file}\n")
        self.path = self.directory + '/' + self.selected_file
        #return selected_file

    def enable_button(self):
        self.start_button.setDisabled(False)

    def enable_button_2(self):
        self.start_button_2.setDisabled(False)

    def calculate_opt_value(self):
        if self.qrb1.isChecked():
            self.my_original_df, self.selected_columns, self.mydf4clust, self.mythresholds, self.myscores, self.myopt_thresold = myHAC.calculate_threshold(self.path)
            self.recommended_text.setPlainText(str(float('%.3f'%(self.myopt_thresold))))

        else:
            self.my_original_df, self.selected_columns, self.mydf4clust, self.myepsilons, self.myscores, self.myopt_epsilon = myDBScan.calculate_epsilon(self.path)
            self.recommended_text.setPlainText(str(float('%.3f'%(self.myopt_epsilon))))

    def draw_silhouette(self):
        # Clear the figure
        self.fig_silhouette.clear()

        # Create a subplot
        ax = self.fig_silhouette.add_subplot(111)
        ax.set_ylabel('Silhouette score')

        if self.qrb1.isChecked():
            ax.scatter(self.mythresholds, self.myscores, c='k', marker='+')
            ax.axvline(x=self.myopt_thresold, c='r', ls=':')
            ax.set_xlabel('Distance threshold')
        else:
            ax.scatter(self.myepsilons, self.myscores, c='k', marker='+')
            ax.axvline(x=self.myopt_epsilon, c='r', ls=':')
            ax.set_xlabel('Epsilon')

        self.fig_silhouette.tight_layout()
        # Refresh the canvas
        self.canvas.draw()

    def start_clustering(self):
        if self.qrb1.isChecked():
            self.final_df, self.Cluster_nb, self.Cluster_part_nb, self.colors = myHAC.clustering(self.path, self.my_original_df, self.selected_columns, self.mydf4clust, float(self.set_text.text().replace(',','.'))) # colormap = 'viridis') # automatically replace the float separator in case it is a comma (keyboard dependant)
        else:
            self.final_df, self.Cluster_nb, self.Cluster_part_nb, self.colors = myDBScan.clustering(self.path, self.my_original_df, self.selected_columns, self.mydf4clust, float(self.set_text.text().replace(',','.'))) # colormap = 'viridis') # automatically replace the float separator in case it is a comma

    def draw_hbar(self):
        # ensure good representation/distribution of graphs
        self.fig_horizbar.clear()

        # extract isotope from the selected columns
        self.isotopes = []
        for char in self.selected_columns:
            self.isotopes = [char.split(' ', maxsplit=1)[0] for char in self.selected_columns]
            self.x_label = char.split(' ', maxsplit=1)[1]

        #ax = ax.ravel()
        print('\n * The average composition per cluster is:')

        for k in self.Cluster_nb:
            print('Cluster' + str(k) + ' -->')
            # create lists for mean and sd of cluster compo
            list_mean = []
            list_sd = []
            for l, m in enumerate(self.selected_columns):
                mean = [np.mean(self.final_df[m][self.final_df['Cluster'] == k])]
                sd = [np.std(self.final_df[m][self.final_df['Cluster'] == k])]
                list_mean.append(mean)
                list_sd.append(sd)

                print('- ' + str(m) +': {:.1f} %'.format(mean[-1]*100)+' +/- {:.1f} %'.format(sd[-1]*100))

            list_mean = list(np.concatenate(list_mean).flat)
            list_sd = list(np.concatenate(list_sd).flat)

            ax = self.fig_horizbar.add_subplot(1, len(self.Cluster_nb), k+1)#, sharex=True, sharey=True)

            ax.barh(np.arange(len(list_mean)), list_mean, xerr=list_sd, align='center', color=self.colors[k])
            ax.set_yticks(np.arange(len(self.isotopes)))
            ax.set_yticklabels(self.isotopes)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel(self.x_label)
            ax.set_xlim(-0.1,1.1)
            ax.set_title('Cluster n°'+str(k)+'\n('+str(self.Cluster_part_nb[k])+' part.)', fontsize=10)

            print('\n')
            # clear lists
            list_mean.clear()
            list_sd.clear()

        #plt.suptitle(self.kw2+' composition of:', fontsize=12, fontstyle='italic')
        self.fig_horizbar.tight_layout()
        # Refresh the canvas
        self.canvas2.draw()

    def draw_hist(self):
        # ensure good representation/distribution of graphs
        self.fig_mass_distrib.clear()

        # determine histogram limits
        h_min = np.min(self.final_df['total mass [fg]'])
        h_max = np.max(self.final_df['total mass [fg]'])
        h_range = (h_min, h_max)
        h_bins = int(np.sqrt(self.final_df['total mass [fg]'].shape[0]))

        print('* The average total mass per cluster is:')

        # plot the histogram
        for k in self.Cluster_nb:
            ax = self.fig_mass_distrib.add_subplot(1, len(self.Cluster_nb), k+1)#, sharex=True, sharey=True)

            ax.hist(self.final_df['total mass [fg]'][self.final_df['Cluster'] == k], bins=h_bins, range=h_range, density=True, color=self.colors[k])
            ax.set_title('Cluster n°'+str(k)+'\n('+str(self.Cluster_part_nb[k])+' part.)', fontsize=10)
            mean = np.mean(self.final_df['total mass [fg]'][self.final_df['Cluster'] == k])
            sd = np.std(self.final_df['total mass [fg]'][self.final_df['Cluster'] == k])
            ax.annotate('$\overline{m}$ '+'= {:.1f} fg'.format(mean)+'\n$\pm$ '+'{:.1f} fg'.format(sd), xy=(0.5, 0.7), xycoords='axes fraction', fontweight='bold')
            ax.set_xlabel('total mass [fg]')
            if k==0:
                ax.set_ylabel('Density')

            print('Cluster' + str(k) + ' --> {:.3f} fg'.format(mean)+' +/- {:.3f} fg'.format(sd))

        #print(self.df)
        print('\n')
        print('* The total mass of NPs in the sample is ' + '{:.1f} fg'.format(np.sum(self.final_df['total mass [fg]'])))
        #plt.suptitle('Particle mass [fg] distribution of:', fontsize=12, fontstyle='italic')

        self.fig_mass_distrib.tight_layout()
        # Refresh the canvas
        self.canvas3.draw()


# =============================================================
# =============================================================

def main():
    app = QApplication([])
    file_browser = MainWindow()
    file_browser.show()
    app.exec_()


if __name__ == '__main__':
    main()
