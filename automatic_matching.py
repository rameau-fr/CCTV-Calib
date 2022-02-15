from Designer_GUI.automatic_matching_ui import Ui_Automatic_matching_Dialog
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QPixmap, QImage   
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtWidgets import QMessageBox

import cv2
import numpy as np
import random
import math

import pdb

#image pre-processing
from Calibration.color_transfer import color_transfer

#pnplib
from Calibration.pnp_lib import *

#import opencv headless (due to some conflivt)
'''
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
'''

#superglue related imports
import torch
from models_superglue.matching import Matching
from models_superglue.utils import ( frame2tensor,process_resize)
torch.set_grad_enabled(False)

def process_image(image_rgb, device, resize, rotation, resize_float):
    image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales


def image_preprocessing(image_sat, image_cctv, color_align_flag, denoise_sat_flag,  denoise_cctv_flag):
     # Preprocessing!! make library for that
    if color_align_flag==True:
        print('color alignment')
        #color transfer
        image_sat_c = color_transfer(image_cctv, image_sat, clip=True, preserve_paper=False)
    else:
        image_sat_c = image_sat

    #Denoising
    if (denoise_sat_flag==True):
        print('satellite image denoising')
        image_sat_d = cv2.fastNlMeansDenoisingColored(image_sat_c,None,10,10,7,21)
    else:
        image_sat_d = image_sat_c

    if (denoise_cctv_flag==True):
        print('cctv image denoising')
        image_cctv_d = cv2.fastNlMeansDenoisingColored(image_cctv,None,10,10,7,21)
    else:
        image_cctv_d = image_cctv
    
    return image_sat_d, image_cctv_d


#Main window of the toolbox
class AutomaticMatchingWindow(qtw.QWidget):

    closed = QtCore.pyqtSignal()

    def __init__(self, image_sat_color, image_cctv_color):
        super().__init__()
        self.ui = Ui_Automatic_matching_Dialog()
        self.ui.setupUi(self)

        # parameters
        self.color_align_flag = False
        self.denoise_sat_flag = False
        self.denoise_cctv_flag = False

        #open the images
        self.image_sat_color = image_sat_color
        self.image_cctv_color = image_cctv_color
        #self.image_sat_color = cv2.imread('HDMapkakao.png')
        #self.image_cctv_color = cv2.imread('Cam_001.jpg')

        #initialize display
        self.image_sat = self.image_sat_color.copy()
        self.image_cctv = self.image_cctv_color.copy()
        self.crop_sat_img = self.image_sat.copy()
        self.crop_cctv_img = self.image_cctv.copy()
        self.setSatImage()
        self.setCCTVImage()
        self.setCropSatImage()
        self.setCropCCTVImage()

        #rotation parameters
        self.rot_angle = 0
        self.angle_step = 5
        self.rot_mat = np.array([[1,0,0],[0,1,0]])

        #Display parameters
        self.disp_lines = False

        #crop initialize
        self.rect_crop_sat = np.array([0,0,self.image_sat.shape[1],self.image_sat.shape[0]])
        self.rect_crop_cctv = np.array([0,0,self.image_cctv.shape[1],self.image_cctv.shape[0]])

        #Ransac parameters
        self.ransac_nb_it = 10000
        self.ransac_thresh = 10
        self.ransac_model = 3

        #matched keypoints
        self.matched_kpts0 = []
        self.matched_kpts1 = []

        #validation flag
        self.validation = False

        #image_sat_d, image_cctv_d = image_preprocessing(self.image_sat_color, self.image_cctv_color, self.color_align_flag, self.denoise_sat_flag,  self.denoise_cctv_flag)
        #self.image_sat_r_gray = cv2.cvtColor(image_sat_d, cv2.COLOR_BGR2GRAY)
        #self.image_cctv_r_gray = cv2.cvtColor(image_cctv_d, cv2.COLOR_BGR2GRAY)

        #parameters ROI selection on satellite
        self._start_sat = QtCore.QPointF()
        self._current_rect_item_sat = qtw.QGraphicsRectItem()
        self._current_rect_item_sat.setBrush(QtGui.QColor(255,50,50, 60))
        self._current_rect_item_sat.setFlag(qtw.QGraphicsItem.ItemIsMovable, True)
        self.ui.sat_im_graphicsView.scene().addItem(self._current_rect_item_sat)
        self.sat_roi = self.ui.sat_im_graphicsView.scene().sceneRect() 

        #parameters ROI selection on CCTV
        self._start_cctv = QtCore.QPointF()
        self._current_rect_item_cctv = qtw.QGraphicsRectItem()
        self._current_rect_item_cctv.setBrush(QtGui.QColor(255,50,50, 60))
        self._current_rect_item_cctv.setFlag(qtw.QGraphicsItem.ItemIsMovable, True)
        self.ui.cctv_im_graphicsView.scene().addItem(self._current_rect_item_cctv)
        self.cctv_roi = self.ui.cctv_im_graphicsView.scene().sceneRect() 

        #parameters zoom and drag
        self.ui.sat_im_graphicsView.wheelEvent = self.scaleSceneSat
        self.ui.cctv_im_graphicsView.wheelEvent = self.scaleSceneCCTV
        self.ui.matching_im_graphicsView.wheelEvent = self.scaleSceneMatching
        self.ui.sat_im_graphicsView.setDragMode(qtw.QGraphicsView.ScrollHandDrag)
        self.ui.sat_im_graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.ui.cctv_im_graphicsView.setDragMode(qtw.QGraphicsView.ScrollHandDrag)
        self.ui.cctv_im_graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.ui.matching_im_graphicsView.setDragMode(qtw.QGraphicsView.ScrollHandDrag)
        self.ui.matching_im_graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CrossCursor))
        self._mousePressed = None
        self.ui.sat_im_graphicsView.mousePressEvent = self.MymousePressEventSat
        self.ui.sat_im_graphicsView.mouseMoveEvent = self.MymouseMoveEventSat
        self.ui.sat_im_graphicsView.mouseReleaseEvent = self.MymouseReleaseEventSat
        self.ui.cctv_im_graphicsView.mousePressEvent = self.MymousePressEventCCTV
        self.ui.cctv_im_graphicsView.mouseMoveEvent = self.MymouseMoveEventCCTV
        self.ui.cctv_im_graphicsView.mouseReleaseEvent = self.MymouseReleaseEventCCTV
        self.ui.matching_im_graphicsView.mousePressEvent = self.MymousePressEventMatching
        self.ui.matching_im_graphicsView.mouseMoveEvent = self.MymouseMoveEventMatching
        self.ui.matching_im_graphicsView.mouseReleaseEvent = self.MymouseReleaseEventMatching

        #Push button callbacks
        self.ui.rot_left_pushButton.clicked.connect(self.RotateLeft)
        self.ui.rot_right_pushButton.clicked.connect(self.RotateRight)
        self.ui.compute_matching_pushButton.clicked.connect(self.ComputeMatching)

        #checkbox callbacks
        self.ui.color_align_checkBox.stateChanged.connect(self.activColorAlign)
        self.ui.sat_denoise_checkBox.stateChanged.connect(self.activDenoiseSat)
        self.ui.cctv_denoise_checkBox.stateChanged.connect(self.activDenoiseCCTV)
        self.ui.Disp_Lines_checkBox.stateChanged.connect(self.activDispLine)

        self.ui.validate_buttonBox.button(qtw.QDialogButtonBox.Apply).clicked.connect(self.validate_points)
        

        #update the images to fit the windows
        print(self.sat_roi)
        self.updateView()

        #load the superpoint//superglue parameters
        self.nms_radius = 4
        self.keypoint_threshold = 0.005
        self.max_keypoints = 1024
        self.superglue_weights = 'outdoor'
        self.sinkhorn_iterations = 20
        self.match_threshold = 0.2
        self.device = 'cpu' 
        self.config = {
            'superpoint': {
                'nms_radius': self.nms_radius,
                'keypoint_threshold': self.keypoint_threshold,
                'max_keypoints': self.max_keypoints
            },
            'superglue': {
                'weights': self.superglue_weights,
                'sinkhorn_iterations': self.sinkhorn_iterations,
                'match_threshold': self.match_threshold,
            }
        }
        self.matching = Matching(self.config).eval().to(self.device)

        #image preprocessing
        self.resize = [640, 480]
        self.resize_float = True
        self.rot0, self.rot1 = 0, 0

        #display our parameters in the GUI
        self.ui.sp_resize_w_lineEdit.setText(str(self.resize[0]))
        self.ui.sp_resize_h_lineEdit.setText(str(self.resize[1]))
        self.ui.sp_sinkhorn_it_lineEdit.setText(str(self.sinkhorn_iterations))
        self.ui.sp_nb_pts_lineEdit.setText(str(self.max_keypoints))
        self.ui.sp_kp_thresh_lineEdit.setText(str(self.keypoint_threshold))
        self.ui.sp_matching_thresh_lineEdit.setText(str(self.match_threshold))
        self.ui.ransac_thresh_lineEdit.setText(str(self.ransac_thresh))
        self.ui.ransac_it_lineEdit.setText(str(self.ransac_nb_it))
        self.ui.model_comboBox.setCurrentIndex(self.ransac_model-1)

        #check change in text
        self.ui.sp_resize_w_lineEdit.textChanged.connect(self.adjustResizeW)
        self.ui.sp_resize_h_lineEdit.textChanged.connect(self.adjustResizeH)
        self.ui.sp_sinkhorn_it_lineEdit.textChanged.connect(self.adjustSinkhornIt)
        self.ui.sp_nb_pts_lineEdit.textChanged.connect(self.adjustSpNbPts)
        self.ui.sp_kp_thresh_lineEdit.textChanged.connect(self.adjustSpKpThresh)
        self.ui.sp_matching_thresh_lineEdit.textChanged.connect(self.adjustSpMatchingThresh)
        self.ui.ransac_thresh_lineEdit.textChanged.connect(self.adjustRansacThresh)
        self.ui.ransac_it_lineEdit.textChanged.connect(self.adjustRansacIt)
        self.ui.model_comboBox.activated.connect(self.changePnP)

    def closeEvent(self, event):
        super(AutomaticMatchingWindow, self).closeEvent(event)
        self.closed.emit()

    def validate_points(self):
        #print(self.points_sat.shape[0])
        print(self.matched_kpts0.shape[0])
        if (self.matched_kpts0.shape[0]>0):
            print ("points selection done")
            self.validation = True
            self.close()
        
    def adjustResizeW(self,text):
        self.resize[0] = int(text)

    def adjustResizeH(self,text):
        self.resize[1] = int(text)
    
    def adjustSinkhornIt(self,text):
        self.sinkhorn_iterations = int(text)
    
    def adjustSpNbPts(self,text):
        self.max_keypoints = int(text)

    def adjustSpKpThresh(self,text):
        self.keypoint_threshold = float(text)

    def adjustSpMatchingThresh(self,text):
        self.match_threshold = float(text)

    def adjustRansacThresh(self,text):
        self.ransac_thresh = float(text)

    def adjustRansacIt(self,text):
        self.ransac_nb_it = int(text)

    def changePnP(self):
        self.ransac_model = self.ui.model_comboBox.currentIndex()+1
        print(self.ransac_model)

    def activColorAlign(self,int):
        if self.ui.color_align_checkBox.isChecked():
            self.color_align_flag = True
        else:
            self.color_align_flag = False

    def activDenoiseSat(self,int):
        if self.ui.sat_denoise_checkBox.isChecked():
            self.denoise_sat_flag = True
        else:
            self.denoise_sat_flag = False

    def activDenoiseCCTV(self,int):
        if self.ui.cctv_denoise_checkBox.isChecked():
            self.denoise_cctv_flag = True
        else:
            self.denoise_cctv_flag = False

    def activDispLine(self,int):
        if self.ui.Disp_Lines_checkBox.isChecked():
            self.disp_lines = True
        else:
            self.disp_lines = False
        self.Match_display(self.matched_kpts0,self.matched_kpts1)


    def RotateLeft(self):
        #check if some images have been loaded properly
        if(self.image_sat.shape[0]>10):
            #update the rotation parameter
            image = self.image_sat_color
            self.rot_angle = self.rot_angle + self.angle_step
            (h, w) = image.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D((cX, cY), self.rot_angle, 1.0)
            self.rot_mat = M
            rotated = cv2.warpAffine(image, M, (w, h))
            self.image_sat = rotated.copy()
            self.setSatImage()

    def RotateRight(self):
        #check if some images have been loaded properly
        if(self.image_sat.shape[0]>10):
            #update the rotation parameter
            image = self.image_sat_color
            self.rot_angle = self.rot_angle - self.angle_step
            (h, w) = image.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D((cX, cY), self.rot_angle, 1.0)
            self.rot_mat = M
            rotated = cv2.warpAffine(image, M, (w, h))
            print(M)
            self.image_sat = rotated.copy()
            self.setSatImage()

    def ComputeMatching(self):
        
        self.config = {
            'superpoint': {
                'nms_radius': self.nms_radius,
                'keypoint_threshold': self.keypoint_threshold,
                'max_keypoints': self.max_keypoints
            },
            'superglue': {
                'weights': self.superglue_weights,
                'sinkhorn_iterations': self.sinkhorn_iterations,
                'match_threshold': self.match_threshold,
            }
        }
        self.matching = Matching(self.config).eval().to(self.device)

        print("Matching process ongoing")
        image_sat = self.crop_sat_img
        image_cctv = self.crop_cctv_img

        #Preprocessing
        print("Image preprocessing")
        image_sat, image_cctv = image_preprocessing(image_sat, image_cctv, self.color_align_flag, self.denoise_sat_flag,  self.denoise_cctv_flag)

        #superglue matching
        print("Superglue matching")
        image0, inp0, scales0 = process_image(image_sat, self.device, self.resize, self.rot0, self.resize_float)
        image1, inp1, scales1 = process_image(image_cctv, self.device, self.resize, self.rot1, self.resize_float)

        #Run the matching
        pred = self.matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        NbMatches = sum(matches!=-1)

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        #display the matches
        #self.Match_display(mkpts0,mkpts1)

        #update the matches, rescale to original size and derotate
        #rescale
        mkpts0[:,0] = mkpts0[:,0]*(scales0[0])
        mkpts0[:,1] = mkpts0[:,1]*(scales0[1])
        mkpts1[:,0] = mkpts1[:,0]*(scales1[0])
        mkpts1[:,1] = mkpts1[:,1]*(scales1[1])

        #translate the crop
        mkpts0[:,0] = mkpts0[:,0] + self.rect_crop_sat[0]
        mkpts0[:,1] = mkpts0[:,1] + self.rect_crop_sat[1]
        mkpts1[:,0] = mkpts1[:,0] + self.rect_crop_cctv[0]
        mkpts1[:,1] = mkpts1[:,1] + self.rect_crop_cctv[1]

        #rotate with respect to the center of the satellite image
        M = np.vstack((self.rot_mat,np.array([0,0,1])))
        M = np.linalg.inv(M)
        mkpts0 = M@np.c_[ mkpts0, np.ones(mkpts0.shape[0]) ].T
        mkpts0 = mkpts0[0:2,:].T

        #Use RANSAC to remove outliers
        print("RANSAC")
        u0 = self.image_cctv_color.shape[1]/2
        v0 = self.image_cctv_color.shape[0]/2
        R,T,f,disto,inliers =  run_ransac(mkpts0, mkpts1, u0, v0, self.ransac_model, self.ransac_nb_it, self.ransac_thresh)
        K = np.array([[f,0,u0],[0,f,v0],[0,0,1]])
        mkpts1 = mkpts1[inliers,:]
        mkpts0 = mkpts0[inliers,:]

        #display the matches
        self.Match_display(mkpts0,mkpts1)

        #final matches
        self.matched_kpts0 = mkpts0
        self.matched_kpts1 = mkpts1

        print("matching done")

        #display results
        self.ui.result_nb_matches_label.setText(str(np.sum(inliers)))
        self.ui.result_focal_label.setText(str(f))
        self.ui.result_disto_label.setText(str(disto[0]))
        
        
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

    def Match_display(self,mkpts0,mkpts1):
        
        image_sat = self.image_sat_color
        image_cctv = self.image_cctv_color
        image_sat_gray =  cv2.cvtColor(image_sat, cv2.COLOR_BGR2GRAY)
        image_cctv_gray =  cv2.cvtColor(image_cctv, cv2.COLOR_BGR2GRAY)
        output = self.drawMatches(image_sat_gray, mkpts0, image_cctv_gray , mkpts1, self.disp_lines)
        image = QImage(output, output.shape[1], output.shape[0], output.strides[0],QImage.Format_RGB888)
        scene = qtw.QGraphicsScene(self)
        pixmap = QPixmap.fromImage(image)
        item = qtw.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.ui.matching_im_graphicsView.setScene(scene)
        self.ui.matching_im_graphicsView.update()
        self.ui.matching_im_graphicsView.fitInView(self.cctv_roi, QtCore.Qt.KeepAspectRatio)
        self.ui.matching_im_graphicsView.update()

        '''
        image_sat = self.crop_sat_img
        image_cctv = self.crop_cctv_img
        image0, inp0, scales0 = process_image(image_sat, self.device, self.resize, self.rot0, self.resize_float)
        image1, inp1, scales1 = process_image(image_cctv, self.device, self.resize, self.rot1, self.resize_float)

        #plot the line
        output = cv2.hconcat([image0, image1])
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        output = output.astype(np.uint8)
        for i in range(len(mkpts0)):
            left = mkpts0[i,:]
            right = tuple(sum(x) for x in zip(mkpts1[i,:], (image0.shape[1], 0)))
            color1 = (list(np.random.choice(range(256), size=3)))  
            color =[int(color1[0]), int(color1[1]), int(color1[2])]  
            #cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), color, lineType= cv2.LINE_AA)
            output = cv2.circle(output, tuple(map(int, left)), 2, color, 2)
            output = cv2.circle(output, tuple(map(int, right)), 2, color, 2)
        
        image = QImage(output, output.shape[1], output.shape[0], output.strides[0],QImage.Format_RGB888)
        scene = qtw.QGraphicsScene(self)
        pixmap = QPixmap.fromImage(image)
        item = qtw.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.ui.matching_im_graphicsView.setScene(scene)
        self.ui.matching_im_graphicsView.update()
        #FIT TO VIEW!!
        print("here")
        '''

    def setSatImage(self):
            """
            display the satellite image
            """
            image = self.image_sat
            self.image_sat_disp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = QImage(self.image_sat_disp, self.image_sat_disp.shape[1],self.image_sat_disp.shape[0], self.image_sat_disp.strides[0],QImage.Format_RGB888)
            scene = qtw.QGraphicsScene(self)
            pixmap = QPixmap.fromImage(image)
            item = qtw.QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            self.ui.sat_im_graphicsView.setScene(scene)
            self._start_sat = QtCore.QPointF()
            self._current_rect_item_sat = qtw.QGraphicsRectItem()
            self._current_rect_item_sat.setBrush(QtGui.QColor(255,50,50, 60))
            self._current_rect_item_sat.setFlag(qtw.QGraphicsItem.ItemIsMovable, True)
            self.ui.sat_im_graphicsView.scene().addItem(self._current_rect_item_sat)
            self.ui.sat_im_graphicsView.update()

    def setCropSatImage(self):
            """
            display the cropped satellite image
            """
            image = self.crop_sat_img
            self.image_crop_sat_disp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = QImage(self.image_crop_sat_disp, self.image_crop_sat_disp.shape[1],self.image_crop_sat_disp.shape[0], self.image_crop_sat_disp.strides[0],QImage.Format_RGB888)
            scene = qtw.QGraphicsScene(self)
            pixmap = QPixmap.fromImage(image)
            item = qtw.QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            self.ui.cropped_sat_im_graphicsView.setScene(scene)
            self.ui.cropped_sat_im_graphicsView.update()

    def setCropCCTVImage(self):
            """
            display the cropped cctv image
            """
            image = self.crop_cctv_img
            self.image_crop_cctv_disp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = QImage(self.image_crop_cctv_disp, self.image_crop_cctv_disp.shape[1],self.image_crop_cctv_disp.shape[0], self.image_crop_cctv_disp.strides[0],QImage.Format_RGB888)
            scene = qtw.QGraphicsScene(self)
            pixmap = QPixmap.fromImage(image)
            item = qtw.QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            self.ui.cropped_cctv_im_graphicsView.setScene(scene)
            self.ui.cropped_cctv_im_graphicsView.update()

    def MymousePressEventSat(self, event):
        # Right-click to select ROI
        if event.button() == QtCore.Qt.RightButton:
            self._mousePressed = QtCore.Qt.RightButton
            self._start_sat = self.ui.sat_im_graphicsView.mapToScene(event.pos())
            r = QtCore.QRectF(self._start_sat, self._start_sat)
            self._current_rect_item_sat.setRect(r)    

         # Middle-click-to-pan
        if event.button() == QtCore.Qt.MidButton:
            self._mousePressed = QtCore.Qt.MidButton
            self._mousePressedPos = event.pos()
            self.ui.sat_im_graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.ClosedHandCursor))
            self._dragPos = event.pos()     

    def MymouseMoveEventSat(self, event):
        mouse_position = self.ui.sat_im_graphicsView.mapToScene(event.pos())
        self.mouse_position = [mouse_position.x(), mouse_position.y()]
        # Middle-click-to-pan
        if self._mousePressed == QtCore.Qt.MidButton:
            newPos = event.pos()
            diff = newPos - self._dragPos
            self._dragPos = newPos
            self.ui.sat_im_graphicsView.horizontalScrollBar().setValue(self.ui.sat_im_graphicsView.horizontalScrollBar().value() - diff.x())
            self.ui.sat_im_graphicsView.verticalScrollBar().setValue(self.ui.sat_im_graphicsView.verticalScrollBar().value() - diff.y())
        # Right-click to select ROI
        if self._mousePressed == QtCore.Qt.RightButton:
            if self._current_rect_item_sat is not None:
                # delete the plot
                r = QtCore.QRectF(QtCore.QRectF(0, 0, 0, 0)).normalized()
                self._current_rect_item_sat.setRect(r)
                r = QtCore.QRectF(self._start_sat, self.ui.sat_im_graphicsView.mapToScene(event.pos())).normalized()
                self._current_rect_item_sat.setRect(r)

    def MymouseReleaseEventSat(self, event):
        if event.button() == QtCore.Qt.MidButton:
            self.ui.sat_im_graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CrossCursor))
            self._mousePressed = None
            self.ui.sat_im_graphicsView.update()
        # Right-click to select ROI
        if event.button() == QtCore.Qt.RightButton:
            self._mousePressed = None
            #save the ROI
            self.sat_roi = self._current_rect_item_sat.rect()

            #Crop the image image display in the cropped image part
            x_val = int(self.sat_roi.x())
            y_val = int(self.sat_roi.y())
            width_val = int(self.sat_roi.width())
            height_val = int(self.sat_roi.height())
            self.rect_crop_sat = np.array([x_val,y_val,width_val,height_val])
            crop_img = self.image_sat[y_val:y_val+height_val, x_val:x_val+width_val]
            self.crop_sat_img = crop_img
            self.setCroppedSatImage()
            print(self.crop_sat_img.shape)


    def resetViewToROISat(self):
        self.ui.sat_im_graphicsView.fitInView(self.sat_roi, QtCore.Qt.KeepAspectRatio)
        self.ui.sat_im_graphicsView.update()


    def setCCTVImage(self):
        """
        display the current cctv image
        """
        current_image = self.image_cctv
        image = current_image
        self.image_cctv_disp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(self.image_cctv_disp, self.image_cctv_disp.shape[1],self.image_cctv_disp.shape[0], self.image_cctv_disp.strides[0],QImage.Format_RGB888)
        scene = qtw.QGraphicsScene(self)
        pixmap = QPixmap.fromImage(image)
        item = qtw.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.ui.cctv_im_graphicsView.setScene(scene)
        self._start_cctv = QtCore.QPointF()
        self._current_rect_item_cctv = qtw.QGraphicsRectItem()
        self._current_rect_item_cctv.setBrush(QtGui.QColor(255,50,50, 60))
        self._current_rect_item_cctv.setFlag(qtw.QGraphicsItem.ItemIsMovable, True)
        self.ui.cctv_im_graphicsView.scene().addItem(self._current_rect_item_cctv)
        self.cctv_roi = self.ui.cctv_im_graphicsView.scene().sceneRect() 
        self.ui.cctv_im_graphicsView.update()


    def MymousePressEventCCTV(self, event):
         # Middle-click-to-pan
        if event.button() == QtCore.Qt.MidButton:
            # delete the plot
            r = QtCore.QRectF(QtCore.QRectF(0, 0, 0, 0)).normalized()
            self._current_rect_item_cctv.setRect(r)
            self.resetViewToROICCTV()
            self.ui.cctv_im_graphicsView.update()
            #
            self._mousePressed = QtCore.Qt.MidButton
            self._mousePressedPos = event.pos()
            self.ui.cctv_im_graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.ClosedHandCursor))
            self._dragPos = event.pos()     
       
        # Right-click to select ROI
        if event.button() == QtCore.Qt.RightButton:
            self._mousePressed = QtCore.Qt.RightButton
            self._start_cctv = self.ui.cctv_im_graphicsView.mapToScene(event.pos())
            r = QtCore.QRectF(self._start_cctv, self._start_cctv)
            self._current_rect_item_cctv.setRect(r)       

    def MymouseMoveEventCCTV(self, event):
        mouse_position = self.ui.cctv_im_graphicsView.mapToScene(event.pos())
        self.mouse_position = [mouse_position.x(), mouse_position.y()]
        # Middle-click-to-pan
        if self._mousePressed == QtCore.Qt.MidButton:
            newPos = event.pos()
            diff = newPos - self._dragPos
            self._dragPos = newPos
            self.ui.cctv_im_graphicsView.horizontalScrollBar().setValue(self.ui.cctv_im_graphicsView.horizontalScrollBar().value() - diff.x())
            self.ui.cctv_im_graphicsView.verticalScrollBar().setValue(self.ui.cctv_im_graphicsView.verticalScrollBar().value() - diff.y())
        # Right-click to select ROI
        if self._mousePressed == QtCore.Qt.RightButton:
            if self._current_rect_item_cctv is not None:
                r = QtCore.QRectF(self._start_cctv, self.ui.cctv_im_graphicsView.mapToScene(event.pos())).normalized()
                self._current_rect_item_cctv.setRect(r)

    def MymouseReleaseEventCCTV(self, event):
        if event.button() == QtCore.Qt.MidButton:
            self.ui.cctv_im_graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CrossCursor))
            self._mousePressed = None
            self.ui.cctv_im_graphicsView.update()
         # Right-click to select ROI
        if event.button() == QtCore.Qt.RightButton:
            self._mousePressed = None
            #save the ROI
            self.cctv_roi = self._current_rect_item_cctv.rect()
            #Crop the image image display in the cropped image part
            x_val = int(self.cctv_roi.x())
            y_val = int(self.cctv_roi.y())
            width_val = int(self.cctv_roi.width())
            height_val = int(self.cctv_roi.height())
            self.rect_crop_cctv = np.array([x_val,y_val,width_val,height_val])
            crop_img = self.image_cctv[y_val:y_val+height_val, x_val:x_val+width_val]
            self.crop_cctv_img = crop_img
            self.setCroppedCCTVImage()
            print(self.crop_cctv_img.shape)
            

    def resetViewToROICCTV(self):
        self.ui.cctv_im_graphicsView.fitInView(self.cctv_roi, QtCore.Qt.KeepAspectRatio)
        self.ui.cctv_im_graphicsView.update()
            
    


    def updateView(self):
        scene_sat = self.ui.sat_im_graphicsView.scene()
        scene_cctv = self.ui.cctv_im_graphicsView.scene()
        scene_sat_cropped = self.ui.cropped_sat_im_graphicsView.scene()
        scene_cctv_cropped = self.ui.cropped_cctv_im_graphicsView.scene()
        if scene_sat is not None:
            r_sat = scene_sat.sceneRect()
            self.ui.sat_im_graphicsView.fitInView(r_sat, QtCore.Qt.KeepAspectRatio)
            self.ui.sat_im_graphicsView.update()
        if scene_cctv is not None:    
            r_cctv = scene_cctv.sceneRect()
            self.ui.cctv_im_graphicsView.fitInView(r_cctv, QtCore.Qt.KeepAspectRatio)
            self.ui.cctv_im_graphicsView.update()
        if scene_sat_cropped is not None:    
            r_sat_crop = scene_sat_cropped.sceneRect()
            self.ui.cropped_sat_im_graphicsView.fitInView(r_sat_crop, QtCore.Qt.KeepAspectRatio)
            self.ui.cropped_sat_im_graphicsView.update()
        if scene_cctv_cropped is not None:    
            r_cctv_crop = scene_cctv_cropped.sceneRect()
            self.ui.cropped_cctv_im_graphicsView.fitInView(r_cctv_crop, QtCore.Qt.KeepAspectRatio)
            self.ui.cropped_cctv_im_graphicsView.update()
        
    def resizeEvent(self, event):
        self.updateView()

    def showEvent(self, event):
        if not event.spontaneous():
            self.updateView()


    def setCroppedSatImage(self):
        """
        display the cropped image
        """
        image = self.crop_sat_img
        self.image_crop_sat_disp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(self.image_crop_sat_disp, self.image_crop_sat_disp.shape[1],self.image_crop_sat_disp.shape[0], self.image_crop_sat_disp.strides[0],QImage.Format_RGB888)
        scene = qtw.QGraphicsScene(self)
        pixmap = QPixmap.fromImage(image)
        item = qtw.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.ui.cropped_sat_im_graphicsView.setScene(scene)
        self.ui.cropped_sat_im_graphicsView.update()
        scene_sat_cropped = self.ui.cropped_sat_im_graphicsView.scene()
        r_sat_crop = scene_sat_cropped.sceneRect()
        self.ui.cropped_sat_im_graphicsView.fitInView(r_sat_crop, QtCore.Qt.KeepAspectRatio)
        self.ui.cropped_sat_im_graphicsView.update()

    def setCroppedCCTVImage(self):
        """
        display the cropped image
        """
        image = self.crop_cctv_img
        self.image_crop_cctv_disp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(self.image_crop_cctv_disp, self.image_crop_cctv_disp.shape[1],self.image_crop_cctv_disp.shape[0], self.image_crop_cctv_disp.strides[0],QImage.Format_RGB888)
        scene = qtw.QGraphicsScene(self)
        pixmap = QPixmap.fromImage(image)
        item = qtw.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.ui.cropped_cctv_im_graphicsView.setScene(scene)
        self.ui.cropped_cctv_im_graphicsView.update()
        scene_cctv_cropped = self.ui.cropped_cctv_im_graphicsView.scene()
        r_cctv_crop = scene_cctv_cropped.sceneRect()
        self.ui.cropped_cctv_im_graphicsView.fitInView(r_cctv_crop, QtCore.Qt.KeepAspectRatio)
        self.ui.cropped_cctv_im_graphicsView.update()





        


    def scaleSceneSat(self, event):
        self.ui.sat_im_graphicsView.setTransformationAnchor(qtw.QGraphicsView.NoAnchor)
        self.ui.sat_im_graphicsView.setResizeAnchor(qtw.QGraphicsView.NoAnchor)
        oldPos = self.ui.sat_im_graphicsView.mapToScene(event.pos())
        delta = 1.0015**event.angleDelta().y()
        self.ui.sat_im_graphicsView.scale(delta, delta)
        newPos = self.ui.sat_im_graphicsView.mapToScene(event.pos())
        delta = newPos - oldPos
        self.ui.sat_im_graphicsView.translate(delta.x(), delta.y())
        self.ui.sat_im_graphicsView.update()

    def scaleSceneCCTV(self, event):
        self.ui.cctv_im_graphicsView.setTransformationAnchor(qtw.QGraphicsView.NoAnchor)
        self.ui.cctv_im_graphicsView.setResizeAnchor(qtw.QGraphicsView.NoAnchor)
        oldPos = self.ui.cctv_im_graphicsView.mapToScene(event.pos())
        delta = 1.0015**event.angleDelta().y()
        self.ui.cctv_im_graphicsView.scale(delta, delta)
        newPos = self.ui.cctv_im_graphicsView.mapToScene(event.pos())
        delta = newPos - oldPos
        self.ui.cctv_im_graphicsView.translate(delta.x(), delta.y())

    def scaleSceneMatching(self, event):
        self.ui.matching_im_graphicsView.setTransformationAnchor(qtw.QGraphicsView.NoAnchor)
        self.ui.matching_im_graphicsView.setResizeAnchor(qtw.QGraphicsView.NoAnchor)
        oldPos = self.ui.matching_im_graphicsView.mapToScene(event.pos())
        delta = 1.0015**event.angleDelta().y()
        self.ui.matching_im_graphicsView.scale(delta, delta)
        newPos = self.ui.matching_im_graphicsView.mapToScene(event.pos())
        delta = newPos - oldPos
        self.ui.matching_im_graphicsView.translate(delta.x(), delta.y())
        self.ui.matching_im_graphicsView.update()


    def MymousePressEventMatching(self, event):
         # Middle-click-to-pan
        if event.button() == QtCore.Qt.MidButton:
            self._mousePressed = QtCore.Qt.MidButton
            self._mousePressedPos = event.pos()
            self.ui.matching_im_graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.ClosedHandCursor))
            self._dragPos = event.pos()     

    def MymouseMoveEventMatching(self, event):
        mouse_position = self.ui.matching_im_graphicsView.mapToScene(event.pos())
        self.mouse_position = [mouse_position.x(), mouse_position.y()]
        # Middle-click-to-pan
        if self._mousePressed == QtCore.Qt.MidButton:
            newPos = event.pos()
            diff = newPos - self._dragPos
            self._dragPos = newPos
            self.ui.matching_im_graphicsView.horizontalScrollBar().setValue(self.ui.matching_im_graphicsView.horizontalScrollBar().value() - diff.x())
            self.ui.matching_im_graphicsView.verticalScrollBar().setValue(self.ui.matching_im_graphicsView.verticalScrollBar().value() - diff.y())

    def MymouseReleaseEventMatching(self, event):
        if event.button() == QtCore.Qt.MidButton:
            self.ui.matching_im_graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CrossCursor))
            self._mousePressed = None
            self.ui.matching_im_graphicsView.update()


'''
if __name__ == "__main__":
    app = qtw.QApplication([])
    widget = AutomaticMatchingWindow()
    widget.show()
    app.exec_()
'''