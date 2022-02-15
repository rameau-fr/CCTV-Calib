#main window ui
from Designer_GUI.cctv_calib_ui import Ui_Main_Dialog
from kp_selection import PointClickWindow
from binary_thresh_select import BinaryThreshWindow
from disp_overlay import DispOverlayWindow
from google_map import GoogleMapWindow
from automatic_matching import AutomaticMatchingWindow

#Pyqt lib
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5.QtGui import QPixmap, QImage   
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QMessageBox

#Other
#import cv2
import numpy as np
import random

#out library for calibration
from Calibration.calibration_lib import *
from Calibration.disto_div_lib import *
from Calibration.video_to_median_image import *

import pdb

#import opencv headless (due to some conflivt)
import os, sys
ci_build_and_not_headless = False
try:
    from cv2.version import ci_build, headless
    ci_and_not_headless = ci_build and not headless
except:
    pass
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_FONTDIR")
import cv2


# CCTV class
class CCTVCamera():
    def __init__(self):
        
        # images
        self.image_path = ""
        self.image = np.array([[0,0],[0,0]])
        self.image_cctv_rec = np.array([[0,0],[0,0]])
        self.image_display = np.array([[0,0],[0,0]])
        self.cam_id = 0
        self.color = (0,0,0)
        self.cctv_pts = np.array([])
        self.sat_pts = np.array([])

        #initialize parameters for this camera
        self.calib_params = parameters_calibration()
        self.display_params = parameters_display()

        #camera calibration results
        self.Elevation = []
        self.K = []
        self.dist = []
        self.R = []
        self.T = []
        self.K_rec = []
        self.Hsat2cctv = []
        self.Hsat2cctv_rec = []
        self.hori_line = []
        self.N = []
        self.cam_center_sat = []
        self.cam_center_gps = []
        self.coverage_sat = []
        self.coverage_gps = []
        self.K_pre_cal = []
        self.disto_pre_cal = []



#Main window of the toolbox
class MainWindow(qtw.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_Main_Dialog()
        self.ui.setupUi(self)

        #Project Name 
        self.ProjectName = "MyProject"
        #button click
        self.ui.load_sat_im_button.clicked.connect(self.loadSatImage)
        self.ui.add_camera_button.clicked.connect(self.addNewCCTV)
        self.ui.select_keypoints_button.clicked.connect(self.SelectKeypoints)
        self.ui.create_from_GM_button.clicked.connect(self.RunGoogleMapSat)
        self.ui.load_sat_pts_button.clicked.connect(self.loadGPSSatPts)
        self.ui.calib_cam_keypoints_button.clicked.connect(self.calibrateCurrentCamera)
        self.ui.set_bin_thresh_pushButton.clicked.connect(self.SelectBinaryThresh)
        self.ui.display_overlap_button.clicked.connect(self.DisplayOverlay)
        self.ui.load_int_pushButton.clicked.connect(self.LoadIntrinsic)
        self.ui.save_results_button.clicked.connect(self.SaveResults)
        self.ui.automatic_keypoints_button.clicked.connect(self.AutomaticKeypoints)
        

        #image sat
        self.image_sat = np.array([[0,0],[0,0]])
        self.image_sat_display = np.array([[0,0],[0,0]])

        # cctv camera list
        self.cctv_id = []
        self.cctv_cameras = []
        self.current_cctv_id = 0

        #initialize parameters gps
        self.T_gps2sat = np.eye(3)
        self.scale = 1

        #display parameters GUI
        self.ui.grid_rot_horizontalSlider.valueChanged.connect(self.adjustGridRot)
        self.ui.grid_trans_x_horizontalSlider.valueChanged.connect(self.adjustGridTransX)
        self.ui.grid_trans_y_horizontalSlider.valueChanged.connect(self.adjustGridTransY)
        self.ui.grid_display_checkBox.stateChanged.connect(self.activGridDisp)
        self.ui.hl_display_checkBox.stateChanged.connect(self.activHlDisp)
        self.ui.zenith_display_checkBox.stateChanged.connect(self.activZenithDisp)
        self.ui.grid_cell_size_lineEdit.textChanged.connect(self.adjustCellSize)
        self.ui.grid_length_lineEdit.textChanged.connect(self.adjustGridLength)
        self.ui.grid_width_lineEdit.textChanged.connect(self.adjustWidthLength)
        self.ui.zenith_length_lineEdit.textChanged.connect(self.adjustZenithLength)
        self.ui.max_cov_lineEdit.textChanged.connect(self.adjustMaxCov)

        #adjust calibration parameters
        self.ui.activate_ecc_checkBox.stateChanged.connect(self.activEcc)
        self.ui.ecc_disto_checkBox.stateChanged.connect(self.activEccDisto)
        self.ui.binary_ecc_checkBox.stateChanged.connect(self.activEccBin)
        self.ui.keypoints_activ_checkBox.stateChanged.connect(self.activMatching)
        self.ui.activ_optim_checkBox.stateChanged.connect(self.activOptim)
        self.ui.refine_int_checkBox.stateChanged.connect(self.activRefInt)
        self.ui.color_align_checkBox.stateChanged.connect(self.activColorAlign)
        self.ui.cctv_denoise_checkBox.stateChanged.connect(self.activCCTVDenoise)
        self.ui.sat_denoise_checkBox.stateChanged.connect(self.activSatDenoise)
        self.ui.known_intrinsics_checkBox.stateChanged.connect(self.activKwonInt)
        self.ui.ecc_nb_it_lineEdit.textChanged.connect(self.adjustEccIt)
        self.ui.ecc_nb_scale_lineEdit.textChanged.connect(self.adjustEccScale)
        self.ui.ransac_thresh_lineEdit.textChanged.connect(self.adjustRansacTresh)
        self.ui.ransac_it_lineEdit.textChanged.connect(self.adjustRansacIt)
        self.ui.PnP_comboBox.activated.connect(self.changePnP)
        self.ui.matching_algo_comboBox.activated.connect(self.changeMatching)

        self.ui.project_name_lineEdit.textChanged.connect(self.changeProjectName)

        #parameters zoom and drag Satellite
        self.ui.sat_image_graphicsView.wheelEvent = self.scaleSceneSat
        self.ui.sat_image_graphicsView.setDragMode(qtw.QGraphicsView.ScrollHandDrag)
        self.ui.sat_image_graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self._mousePressed = None
        self.ui.sat_image_graphicsView.mousePressEvent = self.MymousePressEventSat
        self.ui.sat_image_graphicsView.mouseMoveEvent = self.MymouseMoveEventSat
        self.ui.sat_image_graphicsView.mouseReleaseEvent = self.MymouseReleaseEventSat

        #parameters zoom and drag CCTV
        self.ui.cctv_image_graphicsView.wheelEvent = self.scaleSceneCCTV
        self.ui.cctv_image_graphicsView.mousePressEvent = self.MymousePressEventCCTV
        self.ui.cctv_image_graphicsView.mouseMoveEvent = self.MymouseMoveEventCCTV
        self.ui.cctv_image_graphicsView.mouseReleaseEvent = self.MymouseReleaseEventCCTV

        #camera selector
        self.ui.camera_list_widget.itemSelectionChanged.connect(self.CameraselectionChanged)

    ############### Calibration options #############
    def changePnP(self):
        self.cctv_cameras[self.current_cctv_id].calib_params.pnp_method = self.ui.PnP_comboBox.currentIndex()+1

    def changeMatching(self):
        self.cctv_cameras[self.current_cctv_id].calib_params.Matching_type = self.ui.matching_algo_comboBox.currentIndex()+2

    def adjustEccIt(self,text):
        self.cctv_cameras[self.current_cctv_id].calib_params.nb_it = float(text)

    def changeProjectName(self,text):
        self.ProjectName = text

    def adjustEccScale(self,text):
        self.cctv_cameras[self.current_cctv_id].calib_params.nb_scale = float(text)

    def adjustRansacTresh(self,text):
        self.cctv_cameras[self.current_cctv_id].calib_params.ransac_thresh = float(text)

    def adjustRansacIt(self,text):
        self.cctv_cameras[self.current_cctv_id].calib_params.ransac_it = float(text)

    def activEcc(self,int):
        if self.ui.activate_ecc_checkBox.isChecked():
            self.cctv_cameras[self.current_cctv_id].calib_params.dense_registration_flag = True
        else:
            self.cctv_cameras[self.current_cctv_id].calib_params.dense_registration_flag = False

    def activEccDisto(self,int):
        if self.ui.ecc_disto_checkBox.isChecked():
            self.cctv_cameras[self.current_cctv_id].calib_params.ref_disto_flag = True
        else:
            self.cctv_cameras[self.current_cctv_id].calib_params.ref_disto_flag = False

    def activEccBin(self,int):
        if self.ui.binary_ecc_checkBox.isChecked():
            self.cctv_cameras[self.current_cctv_id].calib_params.binary_flag = True
        else:
            self.cctv_cameras[self.current_cctv_id].calib_params.binary_flag = False

    def activMatching(self,int):
        if self.ui.keypoints_activ_checkBox.isChecked():
            self.cctv_cameras[self.current_cctv_id].calib_params.sparse_registration_flag = True
        else:
            self.cctv_cameras[self.current_cctv_id].calib_params.sparse_registration_flag = False

    def activOptim(self,int):
        if self.ui.activ_optim_checkBox.isChecked():
            self.cctv_cameras[self.current_cctv_id].calib_params.non_lin_ref_flag = True
        else:
            self.cctv_cameras[self.current_cctv_id].calib_params.non_lin_ref_flag = False

    def activRefInt(self,int):
        if self.ui.refine_int_checkBox.isChecked():
            self.cctv_cameras[self.current_cctv_id].calib_params.optimize_intrinsic = True
        else:
            self.cctv_cameras[self.current_cctv_id].calib_params.optimize_intrinsic = False

    def activColorAlign(self,int):
        if self.ui.color_align_checkBox.isChecked():
            self.cctv_cameras[self.current_cctv_id].calib_params.color_align_flag = True
        else:
            self.cctv_cameras[self.current_cctv_id].calib_params.color_align_flag = False

    def activCCTVDenoise(self,int):
        if self.ui.cctv_denoise_checkBox.isChecked():
            self.cctv_cameras[self.current_cctv_id].calib_params.denoise_cctv_flag = True
        else:
            self.cctv_cameras[self.current_cctv_id].calib_params.denoise_cctv_flag = False

    def activSatDenoise(self,int):
        if self.ui.sat_denoise_checkBox.isChecked():
            self.cctv_cameras[self.current_cctv_id].calib_params.denoise_sat_flag = True
        else:
            self.cctv_cameras[self.current_cctv_id].calib_params.denoise_sat_flag = False

    def activKwonInt(self,int):
        if self.ui.known_intrinsics_checkBox.isChecked():
            self.cctv_cameras[self.current_cctv_id].calib_params.known_Intrinsic_flag = True
        else:
            self.cctv_cameras[self.current_cctv_id].calib_params.known_Intrinsic_flag = False
    


    ############### CCTV Display options #############
    def adjustMaxCov(self,text):
        self.cctv_cameras[self.current_cctv_id].calib_params.max_dist = float(text)
        self.cctv_cameras[self.current_cctv_id].image_cctv_rec, self.cctv_cameras[self.current_cctv_id].K_rec, \
            self.cctv_cameras[self.current_cctv_id].Hsat2cctv_rec, self.cctv_cameras[self.current_cctv_id].hori_line, \
            self.cctv_cameras[self.current_cctv_id].N, self.cctv_cameras[self.current_cctv_id].cam_center_sat,\
            self.cctv_cameras[self.current_cctv_id].cam_center_gps, self.cctv_cameras[self.current_cctv_id].Elevation, \
            self.cctv_cameras[self.current_cctv_id].coverage_sat, self.cctv_cameras[self.current_cctv_id].coverage_gps\
            = compute_parameters(self.cctv_cameras[self.current_cctv_id].image, self.cctv_cameras[self.current_cctv_id].K, \
            self.cctv_cameras[self.current_cctv_id].dist,self.cctv_cameras[self.current_cctv_id].R, \
            self.cctv_cameras[self.current_cctv_id].T, self.T_gps2sat, self.scale, self.cctv_cameras[self.current_cctv_id].calib_params)
            
        image_disp = self.image_sat.copy()
        for id_cam in range(len(self.cctv_cameras)):
            image_disp = draw_sat_cam(image_disp, self.cctv_cameras[id_cam].image_cctv_rec, \
                self.cctv_cameras[id_cam].R, self.cctv_cameras[id_cam].T, \
                self.scale, self.cctv_cameras[id_cam].K_rec, self.cctv_cameras[id_cam].coverage_sat, \
                self.cctv_cameras[id_cam].cam_center_sat, self.cctv_cameras[id_cam].display_params, self.cctv_cameras[id_cam].color)
        self.image_sat_display = image_disp.copy()
        self.setSatImage()
        self.resetViewToROISat()

    def adjustZenithLength(self,text):
        self.cctv_cameras[self.current_cctv_id].display_params.zenith_length = float(text)
        self.update_cctv_drawing()

    def adjustGridLength(self, text):
        self.cctv_cameras[self.current_cctv_id].display_params.grid_width = float(text)
        self.update_cctv_drawing()
    
    def adjustWidthLength(self, text):
        self.cctv_cameras[self.current_cctv_id].display_params.grid_length = float(text)
        self.update_cctv_drawing()

    def adjustCellSize(self, text):
        self.cctv_cameras[self.current_cctv_id].display_params.cell_size = float(text)
        self.update_cctv_drawing()

    def activZenithDisp(self,int):
        if self.ui.zenith_display_checkBox.isChecked():
            self.cctv_cameras[self.current_cctv_id].display_params.plot_zenith_flag = True
        else:
            self.cctv_cameras[self.current_cctv_id].display_params.plot_zenith_flag = False
        self.update_cctv_drawing()

    def activHlDisp(self,int):
        if self.ui.hl_display_checkBox.isChecked():
            self.cctv_cameras[self.current_cctv_id].display_params.plot_hor_line_flag = True
        else:
            self.cctv_cameras[self.current_cctv_id].display_params.plot_hor_line_flag = False
        self.update_cctv_drawing()

    def activGridDisp(self,int):
        if self.ui.grid_display_checkBox.isChecked():
            self.cctv_cameras[self.current_cctv_id].display_params.plot_grid_flag = True
        else:
            self.cctv_cameras[self.current_cctv_id].display_params.plot_grid_flag = False
        self.update_cctv_drawing()

    def adjustGridRot(self,value):
        self.cctv_cameras[self.current_cctv_id].display_params.grid_rotation = value
        self.update_cctv_drawing()

    def adjustGridTransX(self,value):
        self.cctv_cameras[self.current_cctv_id].display_params.grid_translation_x = value
        self.update_cctv_drawing()
    
    def adjustGridTransY(self,value):
        self.cctv_cameras[self.current_cctv_id].display_params.grid_translation_y = value
        self.update_cctv_drawing()
       
    def update_cctv_drawing(self):
         if (self.cctv_cameras[self.current_cctv_id].image_cctv_rec.shape[0]>10):
            image_cctv_disp =  draw_cctv_cam(self.cctv_cameras[self.current_cctv_id].image_cctv_rec, \
                        self.cctv_cameras[self.current_cctv_id].K_rec, self.cctv_cameras[self.current_cctv_id].R, \
                        self.cctv_cameras[self.current_cctv_id].T, self.scale, self.cctv_cameras[self.current_cctv_id].Hsat2cctv_rec, \
                        self.cctv_cameras[self.current_cctv_id].hori_line, self.cctv_cameras[self.current_cctv_id].display_params)

            self.cctv_cameras[self.current_cctv_id].image_display = image_cctv_disp.copy()
            self.setCCTVImage()
            self.resetViewToROICCTV()



    ############### Load GPS<->Sat points #############
    def loadGPSSatPts(self):
        """
        load the yml file containing the points
        """
        if (self.image_sat.shape[0]>10):
            filter = "yml(*.yml)"
            filename_gps_sat_pts = QFileDialog.getOpenFileName(filter=filter)[0]
            if filename_gps_sat_pts!="":
                #load file
                fs = cv2.FileStorage(filename_gps_sat_pts, cv2.FILE_STORAGE_READ)
                pts_gps = fs.getNode("pts_gps").mat()
                pts_sat_cal = fs.getNode("pts_sat").mat()
                # Compute the transformation between the gps and the satellite view
                self.T_gps2sat, self.scale = Find_transform_scale_gps_sat(pts_gps,pts_sat_cal,self.image_sat.shape[1])
        else:
            print("Load images first")
            ## add a popuwindow
            QMessageBox.about(self, "Warning", "Please open satellite image first") 


    ############### Run the calibration of the current CCTV camera #############
    def calibrateCurrentCamera(self):

        if (self.image_sat.shape[0]>10 and len(self.cctv_cameras)>0):
            if(self.cctv_cameras[self.current_cctv_id].cctv_pts.shape[0]>0):
                #Run calibration
                pts_cctv = []
                pts_sat = []
                pts_cctv = self.cctv_cameras[self.current_cctv_id].cctv_pts.copy()
                pts_sat = self.cctv_cameras[self.current_cctv_id].sat_pts.copy()
                #print("pts_cctv :: ", pts_cctv)
                #print("pts_sat :: ", pts_sat)
                pts_cctv = np.c_[ pts_cctv, np.ones(pts_cctv.shape[0]) ]
                pts_sat = np.c_[ pts_sat, np.zeros(pts_sat.shape[0]) ]
                image_cctv = self.cctv_cameras[self.current_cctv_id].image
                K = self.cctv_cameras[self.current_cctv_id].K_pre_cal
                dist = self.cctv_cameras[self.current_cctv_id].disto_pre_cal
                params = self.cctv_cameras[self.current_cctv_id].calib_params

                #calibrate the camera
                self.cctv_cameras[self.current_cctv_id].R, self.cctv_cameras[self.current_cctv_id].T, \
                    self.cctv_cameras[self.current_cctv_id].K, self.cctv_cameras[self.current_cctv_id].dist, \
                    self.cctv_cameras[self.current_cctv_id].Hsat2cctv, \
                    overlay_im, pts_sat_cal, pts_cctv_cal = Run_calibration(self.image_sat, image_cctv, pts_sat, pts_cctv, params, K, dist)
                #print(pts_sat_cal[:,0:2])
                self.cctv_cameras[self.current_cctv_id].cctv_pts = pts_cctv_cal
                self.cctv_cameras[self.current_cctv_id].sat_pts = pts_sat_cal[:,0:2]
                #Compute the camera parameters
                self.cctv_cameras[self.current_cctv_id].image_cctv_rec, self.cctv_cameras[self.current_cctv_id].K_rec, \
                    self.cctv_cameras[self.current_cctv_id].Hsat2cctv_rec, self.cctv_cameras[self.current_cctv_id].hori_line, \
                    self.cctv_cameras[self.current_cctv_id].N, self.cctv_cameras[self.current_cctv_id].cam_center_sat,\
                    self.cctv_cameras[self.current_cctv_id].cam_center_gps, self.cctv_cameras[self.current_cctv_id].Elevation, \
                    self.cctv_cameras[self.current_cctv_id].coverage_sat, self.cctv_cameras[self.current_cctv_id].coverage_gps\
                    = compute_parameters(image_cctv, self.cctv_cameras[self.current_cctv_id].K, \
                    self.cctv_cameras[self.current_cctv_id].dist,self.cctv_cameras[self.current_cctv_id].R, \
                    self.cctv_cameras[self.current_cctv_id].T, self.T_gps2sat, self.scale, self.cctv_cameras[self.current_cctv_id].calib_params)

                #print results
                print("----- Calibration results -----")
                print("focal length:", self.cctv_cameras[self.current_cctv_id].K[0,0])
                print("pp:", [self.cctv_cameras[self.current_cctv_id].K[0,2], self.cctv_cameras[self.current_cctv_id].K[1,2]])
                print("camera gps position:", self.cctv_cameras[self.current_cctv_id].cam_center_gps[0:2])
                print("elevation:", self.cctv_cameras[self.current_cctv_id].Elevation)

                #update the display for the satellite image
                image_disp = self.image_sat.copy()
                for id_cam in range(len(self.cctv_cameras)):
                    image_disp = draw_sat_cam(image_disp, self.cctv_cameras[id_cam].image_cctv_rec, \
                        self.cctv_cameras[id_cam].R, self.cctv_cameras[id_cam].T, \
                        self.scale, self.cctv_cameras[id_cam].K_rec, self.cctv_cameras[id_cam].coverage_sat, \
                        self.cctv_cameras[id_cam].cam_center_sat, self.cctv_cameras[id_cam].display_params, self.cctv_cameras[id_cam].color)
                self.image_sat_display = image_disp.copy()
                self.setSatImage()
                self.resetViewToROISat()

                #update the display for the current cctv
                image_cctv_disp =  draw_cctv_cam(self.cctv_cameras[self.current_cctv_id].image_cctv_rec, \
                    self.cctv_cameras[self.current_cctv_id].K_rec, self.cctv_cameras[self.current_cctv_id].R, \
                    self.cctv_cameras[self.current_cctv_id].T, self.scale, self.cctv_cameras[self.current_cctv_id].Hsat2cctv_rec, \
                    self.cctv_cameras[self.current_cctv_id].hori_line, self.cctv_cameras[self.current_cctv_id].display_params)

                self.cctv_cameras[self.current_cctv_id].image_display = image_cctv_disp.copy()
                self.setCCTVImage()
                self.resetViewToROICCTV()
                
            else:
                QMessageBox.about(self, "Warning", "Select keypoints first") 
        else:
            QMessageBox.about(self, "Warning", "Open the cctv and sat images first") 

        #cv2.namedWindow("overlay_after_opt", cv2.WINDOW_NORMAL) 
        #cv2.imshow("overlay_after_opt", overlay_im)
        #cv2.waitKey()

    ############### Satellite Image Display#############
    def loadSatImage(self):
        """
        load the satellite image
        """
        filter = "Images(*.png *.jpg *.jpeg)"
        self.filename_sat = QFileDialog.getOpenFileName(filter=filter)[0]
        if self.filename_sat!="":
            self.image_sat = cv2.imread(self.filename_sat)
            self.image_sat_display = self.image_sat.copy()
        if (self.image_sat.shape[0]>10):
            self.setSatImage()
            self.resetViewToROISat()

    def MymousePressEventSat(self, event):
        # Right-click to select ROI
        if event.button() == QtCore.Qt.RightButton:
            self._mousePressed = QtCore.Qt.RightButton
            self._start_sat = self.ui.sat_image_graphicsView.mapToScene(event.pos())
            r = QtCore.QRectF(self._start_sat, self._start_sat)
            self._current_rect_item_sat.setRect(r)    

         # Left-click-to-pan
        if event.button() == QtCore.Qt.LeftButton:
            self._mousePressed = QtCore.Qt.LeftButton
            self._mousePressedPos = event.pos()
            self.ui.sat_image_graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.ClosedHandCursor))
            self._dragPos = event.pos()     

    def MymouseMoveEventSat(self, event):
        mouse_position = self.ui.sat_image_graphicsView.mapToScene(event.pos())
        self.mouse_position = [mouse_position.x(), mouse_position.y()]
        # Left-click-to-pan
        if self._mousePressed == QtCore.Qt.LeftButton:
            newPos = event.pos()
            diff = newPos - self._dragPos
            self._dragPos = newPos
            self.ui.sat_image_graphicsView.horizontalScrollBar().setValue(self.ui.sat_image_graphicsView.horizontalScrollBar().value() - diff.x())
            self.ui.sat_image_graphicsView.verticalScrollBar().setValue(self.ui.sat_image_graphicsView.verticalScrollBar().value() - diff.y())
        # Right-click to select ROI
        if self._mousePressed == QtCore.Qt.RightButton:
            if self._current_rect_item_sat is not None:
                r = QtCore.QRectF(self._start_sat, self.ui.sat_image_graphicsView.mapToScene(event.pos())).normalized()
                self._current_rect_item_sat.setRect(r)
            
    def MymouseReleaseEventSat(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.ui.sat_image_graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.ArrowCursor))
            self._mousePressed = None
            self.ui.sat_image_graphicsView.update()
        # Right-click to select ROI
        if event.button() == QtCore.Qt.RightButton:
            self._mousePressed = None
            #save the ROI
            self.sat_roi = self._current_rect_item_sat.rect()
            # delete the plot
            r = QtCore.QRectF(QtCore.QRectF(0, 0, 0, 0)).normalized()
            self._current_rect_item_sat.setRect(r)
            self.resetViewToROISat()
            self.ui.sat_image_graphicsView.update()
            
    def scaleSceneSat(self, event):
        self.ui.sat_image_graphicsView.setTransformationAnchor(qtw.QGraphicsView.NoAnchor)
        self.ui.sat_image_graphicsView.setResizeAnchor(qtw.QGraphicsView.NoAnchor)
        oldPos = self.ui.sat_image_graphicsView.mapToScene(event.pos())
        delta = 1.0015**event.angleDelta().y()
        self.ui.sat_image_graphicsView.scale(delta, delta)
        newPos = self.ui.sat_image_graphicsView.mapToScene(event.pos())
        delta = newPos - oldPos
        self.ui.sat_image_graphicsView.translate(delta.x(), delta.y())
        self.ui.sat_image_graphicsView.update()
    
    def resetViewToROISat(self):
        self.ui.sat_image_graphicsView.fitInView(self.sat_roi, QtCore.Qt.KeepAspectRatio)
        self.ui.sat_image_graphicsView.update()

    def updateView(self):
        scene_sat = self.ui.sat_image_graphicsView.scene()
        scene_cctv = self.ui.cctv_image_graphicsView.scene()
        if scene_sat is not None:
            r_sat = scene_sat.sceneRect()
            self.ui.sat_image_graphicsView.fitInView(r_sat, QtCore.Qt.KeepAspectRatio)
        if scene_cctv is not None:    
            r_cctv = scene_cctv.sceneRect()
            self.ui.cctv_image_graphicsView.fitInView(r_cctv, QtCore.Qt.KeepAspectRatio)
        
    def resizeEvent(self, event):
        self.updateView()

    def showEvent(self, event):
        if not event.spontaneous():
            self.updateView()

    def setSatImage(self):
        """
        display the satellite image
        """
        image = self.image_sat_display
        self.image_sat_disp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(self.image_sat_disp, self.image_sat_disp.shape[1],self.image_sat_disp.shape[0], self.image_sat_disp.strides[0],QImage.Format_RGB888)
        scene = qtw.QGraphicsScene(self)
        pixmap = QPixmap.fromImage(image)
        item = qtw.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.ui.sat_image_graphicsView.setScene(scene)
        self._start_sat = QtCore.QPointF()
        self._current_rect_item_sat = qtw.QGraphicsRectItem()
        self._current_rect_item_sat.setBrush(QtGui.QColor(255,50,50, 60))
        self._current_rect_item_sat.setFlag(qtw.QGraphicsItem.ItemIsMovable, True)
        self.ui.sat_image_graphicsView.scene().addItem(self._current_rect_item_sat)
        self.sat_roi = self.ui.sat_image_graphicsView.scene().sceneRect() 
        self.ui.sat_image_graphicsView.update()

    ############### CCTV Image Display#############
    def addNewCCTV(self):
        """
        Add a new cctv camera
        """

        # Find path
        #filter = "Images(*.png *.jpg *.jpeg)"
        filter = "Images/Video(*.png *.jpg *.jpeg *.mpg *.avi *.mpeg *.MOV *.MP4 *.mp4)"
        filename_cctv = QFileDialog.getOpenFileName(filter=filter)[0]
        if filename_cctv!="":
            print(filename_cctv)
            extension = os.path.splitext(filename_cctv)[1]
            if ((extension == '.png') or (extension == '.jpg') or (extension == '.jpeg')):
                image_cctv = cv2.imread(filename_cctv)
            elif ((extension == '.mpg') or (extension == '.avi') or (extension == '.mpeg') or (extension == '.MOV') or (extension == '.MP4')or (extension == '.mp4')):
                image_cctv = median_image_video(filename_cctv, 20)
            
            # instanciate the class
            currrent_cctv = CCTVCamera()
            currrent_cctv.image_path = filename_cctv
            currrent_cctv.image = image_cctv
            currrent_cctv.image_display = image_cctv.copy()

            #Assign camera id (number)
            current_id = 0
            nb_cam = len(self.cctv_id)
            if (nb_cam>0):
                max_cam_id = max(self.cctv_id)
                flag_insert = 0
                for i in range(0,max_cam_id):
                    if i not in self.cctv_id:
                        current_id = i
                        flag_insert = 1
                        break
                if (flag_insert==0):
                    current_id = max_cam_id + 1
            currrent_cctv.cam_id = current_id
            self.cctv_id.insert(current_id,current_id)
            self.current_cctv_id = current_id

            #automatic color assignment
            color = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
            currrent_cctv.color = color

            # Add the camera to the list of cameras
            self.cctv_cameras.insert(current_id,currrent_cctv)
        
            # Add item in the camera widget
            self.ui.camera_list_widget.addItem("camera_" + str(current_id).zfill(3))

            #update the display parameters in the GUI
            self.ui.grid_display_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].display_params.plot_grid_flag)
            self.ui.hl_display_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].display_params.plot_hor_line_flag)
            self.ui.zenith_display_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].display_params.plot_zenith_flag)
            self.ui.grid_rot_horizontalSlider.setValue(np.int(self.cctv_cameras[self.current_cctv_id].display_params.grid_rotation))
            self.ui.grid_trans_x_horizontalSlider.setValue(np.int(self.cctv_cameras[self.current_cctv_id].display_params.grid_translation_x))
            self.ui.grid_trans_y_horizontalSlider.setValue(np.int(self.cctv_cameras[self.current_cctv_id].display_params.grid_translation_y))
            self.ui.grid_cell_size_lineEdit.setText(str(self.cctv_cameras[self.current_cctv_id].display_params.cell_size))
            self.ui.grid_length_lineEdit.setText(str(self.cctv_cameras[self.current_cctv_id].display_params.grid_length))
            self.ui.grid_width_lineEdit.setText(str(self.cctv_cameras[self.current_cctv_id].display_params.grid_width))
            self.ui.zenith_length_lineEdit.setText(str(self.cctv_cameras[self.current_cctv_id].display_params.zenith_length))
            
            #update the calibration parameters in the GUI
            self.ui.activate_ecc_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.dense_registration_flag)
            self.ui.ecc_disto_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.ref_disto_flag)
            self.ui.binary_ecc_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.binary_flag)
            self.ui.keypoints_activ_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.sparse_registration_flag)
            self.ui.activ_optim_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.non_lin_ref_flag)
            self.ui.refine_int_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.optimize_intrinsic)
            self.ui.color_align_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.color_align_flag)
            self.ui.cctv_denoise_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.denoise_cctv_flag)
            self.ui.sat_denoise_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.denoise_sat_flag)
            self.ui.known_intrinsics_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.known_Intrinsic_flag)
            self.ui.max_cov_lineEdit.setText(str(self.cctv_cameras[self.current_cctv_id].calib_params.max_dist))
            self.ui.ecc_nb_it_lineEdit.setText(str(self.cctv_cameras[self.current_cctv_id].calib_params.nb_it))
            self.ui.ecc_nb_scale_lineEdit.setText(str(self.cctv_cameras[self.current_cctv_id].calib_params.nb_scale))
            self.ui.ransac_thresh_lineEdit.setText(str(self.cctv_cameras[self.current_cctv_id].calib_params.ransac_thresh))
            self.ui.ransac_it_lineEdit.setText(str(self.cctv_cameras[self.current_cctv_id].calib_params.ransac_it))
            self.ui.PnP_comboBox.setCurrentIndex(self.cctv_cameras[self.current_cctv_id].calib_params.pnp_method-1)
            self.ui.matching_algo_comboBox.setCurrentIndex(self.cctv_cameras[self.current_cctv_id].calib_params.Matching_type-2)

            #display the image
            #self.zoom_cctv_im = 1
            self.setCCTVImage()
            self.resetViewToROICCTV()

    def MymousePressEventCCTV(self, event):
         # Left-click-to-pan
        if event.button() == QtCore.Qt.LeftButton:
            self._mousePressed = QtCore.Qt.LeftButton
            self._mousePressedPos = event.pos()
            self.ui.cctv_image_graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.ClosedHandCursor))
            self._dragPos = event.pos()     
        # Right-click to select ROI
        if event.button() == QtCore.Qt.RightButton:
            self._mousePressed = QtCore.Qt.RightButton
            self._start_cctv = self.ui.cctv_image_graphicsView.mapToScene(event.pos())
            r = QtCore.QRectF(self._start_cctv, self._start_cctv)
            self._current_rect_item_cctv.setRect(r)       

    def MymouseMoveEventCCTV(self, event):
        mouse_position = self.ui.cctv_image_graphicsView.mapToScene(event.pos())
        self.mouse_position = [mouse_position.x(), mouse_position.y()]
        # Left-click-to-pan
        if self._mousePressed == QtCore.Qt.LeftButton:
            newPos = event.pos()
            diff = newPos - self._dragPos
            self._dragPos = newPos
            self.ui.cctv_image_graphicsView.horizontalScrollBar().setValue(self.ui.cctv_image_graphicsView.horizontalScrollBar().value() - diff.x())
            self.ui.cctv_image_graphicsView.verticalScrollBar().setValue(self.ui.cctv_image_graphicsView.verticalScrollBar().value() - diff.y())
        # Right-click to select ROI
        if self._mousePressed == QtCore.Qt.RightButton:
            if self._current_rect_item_cctv is not None:
                r = QtCore.QRectF(self._start_cctv, self.ui.cctv_image_graphicsView.mapToScene(event.pos())).normalized()
                self._current_rect_item_cctv.setRect(r)

    def MymouseReleaseEventCCTV(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.ui.cctv_image_graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.ArrowCursor))
            self._mousePressed = None
            self.ui.cctv_image_graphicsView.update()
         # Right-click to select ROI
        if event.button() == QtCore.Qt.RightButton:
            self._mousePressed = None
            #save the ROI
            self.cctv_roi = self._current_rect_item_cctv.rect()
            # delete the plot
            r = QtCore.QRectF(QtCore.QRectF(0, 0, 0, 0)).normalized()
            self._current_rect_item_cctv.setRect(r)
            self.resetViewToROICCTV()
            self.ui.cctv_image_graphicsView.update()

    def resetViewToROICCTV(self):
        self.ui.cctv_image_graphicsView.fitInView(self.cctv_roi, QtCore.Qt.KeepAspectRatio)
        self.ui.cctv_image_graphicsView.update()

    def scaleSceneCCTV(self, event):
        self.ui.cctv_image_graphicsView.setTransformationAnchor(qtw.QGraphicsView.NoAnchor)
        self.ui.cctv_image_graphicsView.setResizeAnchor(qtw.QGraphicsView.NoAnchor)
        oldPos = self.ui.cctv_image_graphicsView.mapToScene(event.pos())
        delta = 1.0015**event.angleDelta().y()
        self.ui.cctv_image_graphicsView.scale(delta, delta)
        newPos = self.ui.cctv_image_graphicsView.mapToScene(event.pos())
        delta = newPos - oldPos
        self.ui.cctv_image_graphicsView.translate(delta.x(), delta.y())
        
    def setCCTVImage(self):
        """
        display the current cctv image
        """
        current_image = self.cctv_cameras[self.current_cctv_id].image_display
        image = current_image
        self.image_cctv_disp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(self.image_cctv_disp, self.image_cctv_disp.shape[1],self.image_cctv_disp.shape[0], self.image_cctv_disp.strides[0],QImage.Format_RGB888)
        scene = qtw.QGraphicsScene(self)
        pixmap = QPixmap.fromImage(image)
        item = qtw.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.ui.cctv_image_graphicsView.setScene(scene)
        self._start_cctv = QtCore.QPointF()
        self._current_rect_item_cctv = qtw.QGraphicsRectItem()
        self._current_rect_item_cctv.setBrush(QtGui.QColor(255,50,50, 60))
        self._current_rect_item_cctv.setFlag(qtw.QGraphicsItem.ItemIsMovable, True)
        self.ui.cctv_image_graphicsView.scene().addItem(self._current_rect_item_cctv)
        self.cctv_roi = self.ui.cctv_image_graphicsView.scene().sceneRect() 
        self.ui.cctv_image_graphicsView.update()

    def CameraselectionChanged(self):
        cam_select_text = self.ui.camera_list_widget.currentItem().text()
        print("Selected items: ", self.ui.camera_list_widget.currentItem().text())
        cam_select = int(cam_select_text[-3:])
        self.current_cctv_id = cam_select
        self.setCCTVImage()
        self.resetViewToROICCTV()

        #update the display parameters in the GUI
        self.ui.grid_display_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].display_params.plot_grid_flag)
        self.ui.hl_display_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].display_params.plot_hor_line_flag)
        self.ui.zenith_display_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].display_params.plot_zenith_flag)
        self.ui.grid_rot_horizontalSlider.setValue(np.int(self.cctv_cameras[self.current_cctv_id].display_params.grid_rotation))
        self.ui.grid_trans_x_horizontalSlider.setValue(np.int(self.cctv_cameras[self.current_cctv_id].display_params.grid_translation_x))
        self.ui.grid_trans_y_horizontalSlider.setValue(np.int(self.cctv_cameras[self.current_cctv_id].display_params.grid_translation_y))
        self.ui.grid_cell_size_lineEdit.setText(str(self.cctv_cameras[self.current_cctv_id].display_params.cell_size))
        self.ui.grid_length_lineEdit.setText(str(self.cctv_cameras[self.current_cctv_id].display_params.grid_length))
        self.ui.grid_width_lineEdit.setText(str(self.cctv_cameras[self.current_cctv_id].display_params.grid_width))
        self.ui.zenith_length_lineEdit.setText(str(self.cctv_cameras[self.current_cctv_id].display_params.zenith_length))
        
        #update the calibration parameters in the GUI
        self.ui.activate_ecc_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.dense_registration_flag)
        self.ui.ecc_disto_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.ref_disto_flag)
        self.ui.binary_ecc_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.binary_flag)
        self.ui.keypoints_activ_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.sparse_registration_flag)
        self.ui.activ_optim_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.non_lin_ref_flag)
        self.ui.refine_int_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.optimize_intrinsic)
        self.ui.color_align_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.color_align_flag)
        self.ui.cctv_denoise_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.denoise_cctv_flag)
        self.ui.sat_denoise_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.denoise_sat_flag)
        self.ui.known_intrinsics_checkBox.setChecked(self.cctv_cameras[self.current_cctv_id].calib_params.known_Intrinsic_flag)
        self.ui.max_cov_lineEdit.setText(str(self.cctv_cameras[self.current_cctv_id].calib_params.max_dist))
        self.ui.ecc_nb_it_lineEdit.setText(str(self.cctv_cameras[self.current_cctv_id].calib_params.nb_it))
        self.ui.ecc_nb_scale_lineEdit.setText(str(self.cctv_cameras[self.current_cctv_id].calib_params.nb_scale))
        self.ui.ransac_thresh_lineEdit.setText(str(self.cctv_cameras[self.current_cctv_id].calib_params.ransac_thresh))
        self.ui.ransac_it_lineEdit.setText(str(self.cctv_cameras[self.current_cctv_id].calib_params.ransac_it))
        self.ui.PnP_comboBox.setCurrentIndex(self.cctv_cameras[self.current_cctv_id].calib_params.pnp_method-1)
        self.ui.matching_algo_comboBox.setCurrentIndex(self.cctv_cameras[self.current_cctv_id].calib_params.Matching_type-2)

    ############### Load Intrinsic #############
    def LoadIntrinsic(self):
        if(self.image_sat.shape[0]>10 and len(self.cctv_cameras)>0 ):
            filter = "yml(*.yml)"
            filename_intrinsic = QFileDialog.getOpenFileName(filter=filter)[0]
            if filename_gps_sat_pts!="":
                #load file
                fs = cv2.FileStorage(filename_gps_sat_pts, cv2.FILE_STORAGE_READ)
                K_pre_cal = fs.getNode("Camera_Matrix").mat()
                disto_pre_cal = fs.getNode("Distortion_Coefficients").mat()

                # store parameters in the struct
                self.cctv_cameras[self.current_cctv_id].K_pre_cal = K_pre_cal
                self.cctv_cameras[self.current_cctv_id].disto_pre_cal = disto_pre_cal

                # change the status to pre-calibrated in GUI and params
                self.ui.known_intrinsics_checkBox.setChecked(True)
                self.cctv_cameras[self.current_cctv_id].calib_params.known_Intrinsic_flag=True
        else:
            print("please load images first")
            QMessageBox.about(self, "Cannot select keypoints", "please load images first")



    ############### Run the overlay GUI #############
    def DisplayOverlay(self):
        #check if some images have been loaded
        if(self.image_sat.shape[0]>10 and len(self.cctv_cameras)>0 ):
            self.widget_bin = DispOverlayWindow(self.image_sat, self.cctv_cameras[self.current_cctv_id].image, self.cctv_cameras[self.current_cctv_id].K, self.cctv_cameras[self.current_cctv_id].R, self.cctv_cameras[self.current_cctv_id].T, self.cctv_cameras[self.current_cctv_id].dist,self.cctv_cameras[self.current_cctv_id].cctv_pts,self.cctv_cameras[self.current_cctv_id].sat_pts)
            self.widget_bin.setWindowModality(QtCore.Qt.ApplicationModal) #wait for the keypoints to be selected
            self.widget_bin.show()

            #block the window until the user has finalize the points selection
            loop = QtCore.QEventLoop()
            self.widget_bin.closed.connect(loop.quit)
            loop.exec_()
        else:
            print("please load images first")
            QMessageBox.about(self, "Cannot select keypoints", "please load images first")


    ############### Run the binary GUI #############
    def SelectBinaryThresh(self):
        #check if some images have been loaded
        if(self.image_sat.shape[0]>10 and len(self.cctv_cameras)>0 ):
            self.widget_bin = BinaryThreshWindow(self.image_sat, self.cctv_cameras[self.current_cctv_id].image)
            self.widget_bin.setWindowModality(QtCore.Qt.ApplicationModal) #wait for the keypoints to be selected
            self.widget_bin.show()

            #block the window until the user has finalize the points selection
            loop = QtCore.QEventLoop()
            self.widget_bin.closed.connect(loop.quit)
            loop.exec_()
            
            #Store the keypoints clicked by the user
            if (self.widget_bin.validation==True):
                self.cctv_cameras[self.current_cctv_id].calib_params.bin_thresh_cctv = self.widget_bin.binary_thresh_cctv
                self.cctv_cameras[self.current_cctv_id].calib_params.bin_thresh_sat = self.widget_bin.binary_thresh_sat
                print('threshold cctv', self.cctv_cameras[self.current_cctv_id].calib_params.bin_thresh_cctv)
                print('threshold sat', self.cctv_cameras[self.current_cctv_id].calib_params.bin_thresh_sat)
        else:
            print("please load images first")
            QMessageBox.about(self, "Cannot select keypoints", "please load images first")


    ############### Run the kp GUI #############
    def SelectKeypoints(self):
        #check if some images have been loaded
        if(self.image_sat.shape[0]>10 and len(self.cctv_cameras)>0 ):
            self.widget_kp = PointClickWindow(self.image_sat, self.cctv_cameras[self.current_cctv_id].image)
            self.widget_kp.setWindowModality(QtCore.Qt.ApplicationModal) #wait for the keypoints to be selected
            self.widget_kp.show()

            #block the window until the user has finalize the points selection
            loop = QtCore.QEventLoop()
            self.widget_kp.closed.connect(loop.quit)
            loop.exec_()
            
            #Store the keypoints clicked by the user
            if (self.widget_kp.validation==True):
                self.cctv_cameras[self.current_cctv_id].cctv_pts = self.widget_kp.points_cctv
                self.cctv_cameras[self.current_cctv_id].sat_pts = self.widget_kp.points_sat
        else:
            print("please load images first")
            QMessageBox.about(self, "Cannot select keypoints", "please load images first")

    ############### Run the automatic match GUI #############
    def AutomaticKeypoints(self):
        #check if some images have been loaded
        if(self.image_sat.shape[0]>10 and len(self.cctv_cameras)>0 ):
            self.widget_auto_kp = AutomaticMatchingWindow(self.image_sat, self.cctv_cameras[self.current_cctv_id].image)
            self.widget_auto_kp.setWindowModality(QtCore.Qt.ApplicationModal) #wait for the keypoints to be selected
            self.widget_auto_kp.show()

            #block the window until the user has finalize the points selection
            loop = QtCore.QEventLoop()
            self.widget_auto_kp.closed.connect(loop.quit)
            loop.exec_()
            
            #Store the keypoints clicked by the user
            if (self.widget_auto_kp.validation==True): 
                self.cctv_cameras[self.current_cctv_id].cctv_pts = self.widget_auto_kp.matched_kpts1
                self.cctv_cameras[self.current_cctv_id].sat_pts = self.widget_auto_kp.matched_kpts0
                print(self.cctv_cameras[self.current_cctv_id].cctv_pts.shape)
        else:
            print("please load images first")
            QMessageBox.about(self, "Cannot select keypoints", "please load images first")

    ############### Run the google map GUI #############
    def RunGoogleMapSat(self):
        self.widget_googlemaps = GoogleMapWindow()
        self.widget_googlemaps.show()

    ######## Function to draw matches #################
    def drawMatches(self,img1, kp1, img2, kp2, dispLines):
        rows1 = img1.shape[0]
        cols1 = img1.shape[1]
        rows2 = img2.shape[0]
        cols2 = img2.shape[1]
        out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
        out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])
        out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])
        for i in range(kp1.shape[0]):
            (x1,y1) = kp1[i]
            (x2,y2) = kp2[i]
            # Draw a small circle at both co-ordinates
            color1 = (list(np.random.choice(range(256), size=3)))  
            color =[int(color1[0]), int(color1[1]), int(color1[2])]  
            cv2.circle(out, (int(x1),int(y1)), 4, color, 4)   
            cv2.circle(out, (int(x2)+cols1,int(y2)), 4, color, 4)
            # Draw a line in between the two points
            if dispLines==True:
                cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), color, 2)

        return out

    ############### Save results #############
    def SaveResults(self):
        # Create the folder to store the result
        isExist = os.path.exists(self.ProjectName)
        if isExist==False:
            os.mkdir(self.ProjectName) 

        #Save the images cctv
        for ind in range(len(self.cctv_cameras)):
            image_cctv = self.cctv_cameras[ind].image_display
            im_ind = str(ind).zfill(4)
            cv2.imwrite(self.ProjectName + '/image_cctv_' + im_ind + '.png', image_cctv)

            #save the warped sat image
            Homog = self.cctv_cameras[ind].K@np.hstack((self.cctv_cameras[ind].R[:,0:2],np.reshape(self.cctv_cameras[ind].T,(3,1))))
            Hw = np.linalg.inv(Homog)
            Hw = Hw/Hw[2,2]
            I_sat_w = spatial_interp_homog_distorsion_Calib(self.image_sat,Hw,cv2.INTER_LINEAR,self.cctv_cameras[ind].image.shape[1],self.cctv_cameras[ind].image.shape[0],self.cctv_cameras[ind].dist,self.cctv_cameras[ind].K[0,2],self.cctv_cameras[ind].K[1,2])
            cv2.imwrite(self.ProjectName + '/image_warped_' + im_ind + '.png', I_sat_w)

            #save matches
            self.cctv_cameras[ind].cctv_pts = self.cctv_cameras[ind].cctv_pts[:,0:2]
            image_sat = self.image_sat
            image_sat_gray =  cv2.cvtColor(image_sat, cv2.COLOR_BGR2GRAY)
            image_cctv_gray =  cv2.cvtColor(self.cctv_cameras[ind].image, cv2.COLOR_BGR2GRAY)
            image_matching = self.drawMatches(image_sat_gray, self.cctv_cameras[ind].sat_pts, image_cctv_gray , self.cctv_cameras[ind].cctv_pts, True)
            cv2.imwrite(self.ProjectName + '/image_matching_' + im_ind + '.png', image_matching)

            #Save matches on warped images
            #Transform the points on the warped satellite image
            W = image_sat.shape[1]; H=image_sat.shape[0]
            u0 = self.cctv_cameras[ind].K[0,2]
            v0 = self.cctv_cameras[ind].K[1,2]
            xx = self.cctv_cameras[ind].sat_pts[:,0]
            yy = self.cctv_cameras[ind].sat_pts[:,1]
            
            nb_pts = xx.shape[0]
            Homog = Homog/Homog[2,2]
            pts_H = Homog@np.vstack((xx, yy, np.ones((1,nb_pts))[0]))
            pts_H = pts_H/np.tile(pts_H[2,:],(3,1))
            p_sat_warp = pts_inverse_division(pts_H.T, self.cctv_cameras[ind].dist, u0, v0)
           
            #draw matches
            I_sat_w_gray =  cv2.cvtColor(I_sat_w, cv2.COLOR_BGR2GRAY)
            image_matching_warp = self.drawMatches(I_sat_w_gray, p_sat_warp, image_cctv_gray , self.cctv_cameras[ind].cctv_pts, True)
            cv2.imwrite(self.ProjectName + '/image_matching_warp_' + im_ind + '.png', image_matching_warp)
            '''
            W = width; H=height
            nx = np.arange(0,W); ny = np.arange(0,H)
            xx, yy = np.meshgrid(nx, ny)

            #apply distortion
            x = xx.flat[:] - u0
            y = yy.flat[:] - v0
            rv = (x**2 + y**2)   
            x_ud = (x/(1+k[0]*rv + k[1]*rv**2)) + u0
            y_ud = (y/(1+k[0]*rv + k[1]*rv**2)) + v0

            # homography
            pts_H = warp@np.vstack((x_ud, y_ud, np.ones((1,H*W))[0]))
            pts_H = pts_H/np.tile(pts_H[2,:],(3,1))

            # interpolate
            pts_H[np.isnan(pts_H)] = 0
            x_ud = pts_H[0,:]; y_ud = pts_H[1,:]
            x_ud = np.reshape(x_ud,(H,W))
            y_ud = np.reshape(y_ud,(H,W))
            I_und = cv2.remap(np.float32(image), np.float32(x_ud), np.float32(y_ud), type_interp)
            '''

            
        #save the satellite image
        image_sat = self.image_sat_display
        cv2.imwrite(self.ProjectName + '/image_sat.png', image_sat)


        # save the yml file containing the parameters
        for ind in range(len(self.cctv_cameras)):
            im_ind = str(ind).zfill(4)
            yml_name = self.ProjectName + "/cam_" + im_ind + ".yml"
            cv_file = cv2.FileStorage(yml_name, cv2.FILE_STORAGE_WRITE)

            W_cctv = self.cctv_cameras[ind].image.shape[1]
            H_cctv = self.cctv_cameras[ind].image.shape[0]
            W_sat = self.image_sat.shape[1]
            H_sat = self.image_sat.shape[0]
            cv_file.write("W_cctv", W_cctv)
            cv_file.write("H_cctv", H_cctv)
            cv_file.write("W_sat", W_sat)
            cv_file.write("H_sat", H_sat)
            H_sat = self.cctv_cameras[ind].image.shape[0]
            if (self.cctv_cameras[ind].calib_params.known_Intrinsic_flag==True):
                cv_file.write("Camera_Matrix", self.cctv_cameras[ind].K_pre_cal)
                cv_file.write("Distortion_Coefficients", self.cctv_cameras[ind].disto_pre_cal)
            else:
                cv_file.write("Camera_Matrix", self.cctv_cameras[ind].K)
                cv_file.write("Distortion_Coefficients", self.cctv_cameras[ind].dist)
            cv_file.write("Hgps2sat", self.T_gps2sat) #gps 2 sat
            cv_file.write("ScaleSat", self.scale) # scale meter per pixel
            cv_file.write("Rotation", self.cctv_cameras[ind].R) # Rotation (pixel)
            cv_file.write("Translation", self.cctv_cameras[ind].T) # Translation (pixel)
            cv_file.write("N", self.cctv_cameras[ind].N) # Translation (pixel)
            cv_file.write("hori_line", self.cctv_cameras[ind].hori_line) # Translation (pixel)
            cv_file.write("cam_center_gps", self.cctv_cameras[ind].cam_center_gps) # gps position cam
            cv_file.write("coverage_sat", self.cctv_cameras[ind].coverage_sat) # coverage cam
            cv_file.write("Elevation", self.cctv_cameras[ind].Elevation) # elevation cam
            cv_file.write("coverage_sat", self.cctv_cameras[ind].coverage_sat) # pixel position cam
            cv_file.write("cam_center_gps", self.cctv_cameras[ind].cam_center_gps) # gps position cam
            cv_file.write("cctv_pts", self.cctv_cameras[ind].cctv_pts) 
            cv_file.write("sat_pts", self.cctv_cameras[ind].sat_pts)
            cv_file.release()

        

    

#run main
if __name__ == "__main__":
    app = qtw.QApplication([])
    widget = MainWindow()
    widget.show()
    app.exec_()