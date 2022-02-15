from Designer_GUI.binary_thresh_ui import Ui_Binary_thresh_Dialog

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
class BinaryThreshWindow(qtw.QWidget):

    closed = QtCore.pyqtSignal()

    def __init__(self, image_sat_color, image_cctv_color):
        super().__init__()
        self.ui = Ui_Binary_thresh_Dialog()
        self.ui.setupUi(self)

        # parameters
        self.binary_thresh_cctv = 180
        self.binary_thresh_sat = 180
        self.color_align_flag = True
        self.denoise_sat_flag = True
        self.denoise_cctv_flag = True

        #open the images
        self.image_sat_color = image_sat_color
        self.image_cctv_color = image_cctv_color
        image_sat_d, image_cctv_d = image_preprocessing(self.image_sat_color, self.image_cctv_color, self.color_align_flag, self.denoise_sat_flag,  self.denoise_cctv_flag)
        self.image_sat_r_gray = cv2.cvtColor(image_sat_d, cv2.COLOR_BGR2GRAY)
        self.image_cctv_r_gray = cv2.cvtColor(image_cctv_d, cv2.COLOR_BGR2GRAY)

        #apply original thresholds
        im_sat_bool = self.image_sat_r_gray > self.binary_thresh_sat
        im_cctv_bool = self.image_cctv_r_gray > self.binary_thresh_cctv
        image_3_channels_cctv = np.zeros(self.image_cctv_color.shape)
        image_3_channels_sat = np.zeros(self.image_sat_color.shape)
        image_3_channels_cctv[:,:,0] = im_cctv_bool*255
        image_3_channels_cctv[:,:,1] = im_cctv_bool*255
        image_3_channels_cctv[:,:,2] = im_cctv_bool*255
        image_3_channels_sat[:,:,0] = im_sat_bool*255
        image_3_channels_sat[:,:,1] = im_sat_bool*255
        image_3_channels_sat[:,:,2] = im_sat_bool*255
        image_sat = image_3_channels_sat
        image_cctv = image_3_channels_cctv
        image_sat = image_sat.astype(np.uint8)
        image_cctv = image_cctv.astype(np.uint8)

        #callback slider
        self.ui.sat_thresh_horizontalSlider.valueChanged.connect(self.adjustSlideThreshSat)
        self.ui.cctv_thresh_horizontalSlider.valueChanged.connect(self.adjustSlideThreshCctv)

        #callback line edit
        self.ui.cctv_thresh_lineEdit.textChanged.connect(self.adjustTreshCctv)
        self.ui.sat_thresh_lineEdit.textChanged.connect(self.adjustTreshSat)

        #callback checkbox
        self.ui.denoise_cctv_checkBox.stateChanged.connect(self.activDenoiseCctv)
        self.ui.denoise_sat_checkBox.stateChanged.connect(self.activDenoiseSat)
        self.ui.align_color_checkBox.stateChanged.connect(self.activColorTransfer)

        #initialize display
        self.image_sat = image_sat
        self.image_cctv = image_cctv
        self.setSatImage()
        self.setCCTVImage()
        
        #parameters zoom and drag
        self.ui.sat_im_graphicsView.wheelEvent = self.scaleSceneSat
        self.ui.cctv_im_graphicsView.wheelEvent = self.scaleSceneCCTV
        self.ui.sat_im_graphicsView.setDragMode(qtw.QGraphicsView.ScrollHandDrag)
        self.ui.sat_im_graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CrossCursor))
        self._mousePressed = None
        self.ui.sat_im_graphicsView.mousePressEvent = self.MymousePressEventSat
        self.ui.sat_im_graphicsView.mouseMoveEvent = self.MymouseMoveEventSat
        self.ui.sat_im_graphicsView.mouseReleaseEvent = self.MymouseReleaseEventSat
        self.ui.cctv_im_graphicsView.mousePressEvent = self.MymousePressEventCCTV
        self.ui.cctv_im_graphicsView.mouseMoveEvent = self.MymouseMoveEventCCTV
        self.ui.cctv_im_graphicsView.mouseReleaseEvent = self.MymouseReleaseEventCCTV
        
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

        #push button
        self.ui.validate_buttonBox.button(qtw.QDialogButtonBox.Apply).clicked.connect(self.validate)

        #validation flag
        self.validation = False
        
    def closeEvent(self, event):
        super(BinaryThreshWindow, self).closeEvent(event)
        self.closed.emit()

    def validate(self):
        self.validation = True
        self.close()

    ############### callback threshold #############

    def activDenoiseCctv(self,int):
        if self.ui.denoise_cctv_checkBox.isChecked():
            self.denoise_cctv_flag = True
        else:
            self.denoise_cctv_flag = False
        self.applyPreProcAndBin()

    def activDenoiseSat(self,int):
        if self.ui.denoise_sat_checkBox.isChecked():
            self.denoise_sat_flag = True
        else:
            self.denoise_sat_flag = False
        self.applyPreProcAndBin()

    def activColorTransfer(self,int):
        if self.ui.align_color_checkBox.isChecked():
            self.color_align_flag = True
        else:
            self.color_align_flag = False
        self.applyPreProcAndBin()
    
    
    def applyPreProcAndBin(self):
        image_sat_d, image_cctv_d = image_preprocessing(self.image_sat_color, self.image_cctv_color, self.color_align_flag, self.denoise_sat_flag,  self.denoise_cctv_flag)
        self.image_sat_r_gray = cv2.cvtColor(image_sat_d, cv2.COLOR_BGR2GRAY)
        self.image_cctv_r_gray = cv2.cvtColor(image_cctv_d, cv2.COLOR_BGR2GRAY)

        #apply original thresholds
        im_sat_bool = self.image_sat_r_gray > self.binary_thresh_sat
        im_cctv_bool = self.image_cctv_r_gray > self.binary_thresh_cctv
        image_3_channels_cctv = np.zeros(self.image_cctv_color.shape)
        image_3_channels_sat = np.zeros(self.image_sat_color.shape)
        image_3_channels_cctv[:,:,0] = im_cctv_bool*255
        image_3_channels_cctv[:,:,1] = im_cctv_bool*255
        image_3_channels_cctv[:,:,2] = im_cctv_bool*255
        image_3_channels_sat[:,:,0] = im_sat_bool*255
        image_3_channels_sat[:,:,1] = im_sat_bool*255
        image_3_channels_sat[:,:,2] = im_sat_bool*255
        image_sat = image_3_channels_sat
        image_cctv = image_3_channels_cctv
        image_sat = image_sat.astype(np.uint8)
        image_cctv = image_cctv.astype(np.uint8)
        self.image_cctv = image_cctv
        self.setCCTVImage()
        self.image_sat = image_sat
        self.setSatImage()


    def adjustSlideThreshSat(self,value):
        self.binary_thresh_sat = value
        self.ui.sat_thresh_lineEdit.setText(str(self.binary_thresh_sat))

    def adjustSlideThreshCctv(self,value):
        self.binary_thresh_cctv = value
        self.ui.cctv_thresh_lineEdit.setText(str(self.binary_thresh_cctv))


    def adjustTreshCctv(self,text):
        self.binary_thresh_cctv = int(text)

        im_cctv_bool = self.image_cctv_r_gray > self.binary_thresh_cctv
        image_3_channels_cctv = np.zeros(self.image_cctv_color.shape)
        image_3_channels_cctv[:,:,0] = im_cctv_bool*255
        image_3_channels_cctv[:,:,1] = im_cctv_bool*255
        image_3_channels_cctv[:,:,2] = im_cctv_bool*255
        image_cctv = image_3_channels_cctv
        image_cctv = image_cctv.astype(np.uint8)

        self.ui.cctv_thresh_horizontalSlider.setValue(self.binary_thresh_cctv)
        
        self.image_cctv = image_cctv
        self.setCCTVImage()

    def adjustTreshSat(self,text):
        self.binary_thresh_sat = int(text)
        im_sat_bool = self.image_sat_r_gray > self.binary_thresh_sat
        image_3_channels_sat = np.zeros(self.image_sat_color.shape)
        image_3_channels_sat[:,:,0] = im_sat_bool*255
        image_3_channels_sat[:,:,1] = im_sat_bool*255
        image_3_channels_sat[:,:,2] = im_sat_bool*255
        image_sat = image_3_channels_sat
        image_sat = image_sat.astype(np.uint8)

        self.ui.sat_thresh_horizontalSlider.setValue(self.binary_thresh_sat)

        self.image_sat = image_sat
        self.setSatImage()


    ############### Satellite Image #############

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
            # delete the plot
            r = QtCore.QRectF(QtCore.QRectF(0, 0, 0, 0)).normalized()
            self._current_rect_item_sat.setRect(r)
            self.resetViewToROISat()
            self.ui.sat_im_graphicsView.update()
            
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
    
    def resetViewToROISat(self):
        self.ui.sat_im_graphicsView.fitInView(self.sat_roi, QtCore.Qt.KeepAspectRatio)
        self.ui.sat_im_graphicsView.update()

    def updateView(self):
        scene_sat = self.ui.sat_im_graphicsView.scene()
        r_sat = scene_sat.sceneRect()
        self.ui.sat_im_graphicsView.fitInView(r_sat, QtCore.Qt.KeepAspectRatio)
        scene_cctv = self.ui.cctv_im_graphicsView.scene()
        r_cctv = scene_cctv.sceneRect()
        self.ui.cctv_im_graphicsView.fitInView(r_cctv, QtCore.Qt.KeepAspectRatio)
        
    def resizeEvent(self, event):
        self.updateView()

    def showEvent(self, event):
        if not event.spontaneous():
            self.updateView()

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
        #self.sat_roi = self.ui.sat_im_graphicsView.scene().sceneRect() 
        self.ui.sat_im_graphicsView.update()

    ############### CCTV Image #############
    def MymousePressEventCCTV(self, event):
         # Middle-click-to-pan
        if event.button() == QtCore.Qt.MidButton:
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
            # delete the plot
            r = QtCore.QRectF(QtCore.QRectF(0, 0, 0, 0)).normalized()
            self._current_rect_item_cctv.setRect(r)
            self.resetViewToROICCTV()
            self.ui.cctv_im_graphicsView.update()

    def resetViewToROICCTV(self):
        self.ui.cctv_im_graphicsView.fitInView(self.cctv_roi, QtCore.Qt.KeepAspectRatio)
        self.ui.cctv_im_graphicsView.update()

    def scaleSceneCCTV(self, event):
        self.ui.cctv_im_graphicsView.setTransformationAnchor(qtw.QGraphicsView.NoAnchor)
        self.ui.cctv_im_graphicsView.setResizeAnchor(qtw.QGraphicsView.NoAnchor)
        oldPos = self.ui.cctv_im_graphicsView.mapToScene(event.pos())
        delta = 1.0015**event.angleDelta().y()
        self.ui.cctv_im_graphicsView.scale(delta, delta)
        newPos = self.ui.cctv_im_graphicsView.mapToScene(event.pos())
        delta = newPos - oldPos
        self.ui.cctv_im_graphicsView.translate(delta.x(), delta.y())
        
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

if __name__ == "__main__":
    app = qtw.QApplication([])
    widget = BinaryThreshWindow()
    widget.show()
    app.exec_()