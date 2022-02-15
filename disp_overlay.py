from Designer_GUI.overlay_disp_ui import Ui_Display_overlay_Dialog

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

#out library for calibration
from Calibration.calibration_lib import *

import pdb


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

#Main window of the toolbox
class DispOverlayWindow(qtw.QWidget):

    closed = QtCore.pyqtSignal()

    def __init__(self, image_sat_color, image_cctv_color, K, R, T, disto, pts_cctv, pts_sat):
        super().__init__()
        self.ui = Ui_Display_overlay_Dialog()
        self.ui.setupUi(self)

        # parameters
        self.alpha = 0.5
        self.R = R
        self.T = T
        self.disto = disto
        self.K = K
        self.pts_sat = pts_sat.astype(int)
        self.pts_cctv = pts_cctv.astype(int)
        self.marker_size = 2
        self.marker_thick = 2 
        self.display_number = False 
        self.display_overlay = True
        self.display_points = True

        #open the images
        self.image_sat_color = image_sat_color
        self.image_cctv_color = image_cctv_color
        
        #apply overlay
        #image_cctv, image_sat = displayOverlay(self.image_cctv_color ,self.image_sat_color,self.K,self.R,self.T,self.disto,self.alpha)
        image_cctv, image_sat = displayOverlay_pts(self.image_cctv_color ,self.image_sat_color,self.K,self.R,self.T,self.disto,self.alpha, self.pts_cctv.astype(int), self.pts_sat.astype(int), self.marker_size, self.marker_thick, self.display_number, self.display_overlay, self.display_points)

        #callback slider
        self.ui.alpha_horizontalSlider.valueChanged.connect(self.adjustAlphaSlide)

        #callback line edit
        self.ui.alpha_lineEdit.textChanged.connect(self.adjustAlphaEdit)

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

        #callbacks to the checkboxes
        self.ui.checkBox_DisplayPoints.stateChanged.connect(self.state_changed_DisplayPoints)
        self.ui.checkBox_DisplayOverlay.stateChanged.connect(self.state_changed_DisplayOverlay)
        self.ui.checkBox_DisplayPtsInd.stateChanged.connect(self.state_changed_DisplayPtsInd)

        #Callbacks spinboxes
        self.ui.spinBox_MarkerThickness.valueChanged.connect(self.getMarkerThickness)
        self.ui.spinBox_MarkerSize.valueChanged.connect(self.getMarkerSize)

        
    def closeEvent(self, event):
        super(DispOverlayWindow, self).closeEvent(event)
        self.closed.emit()

    ############### callback spinboxes #############
    def getMarkerThickness(self):
        self.marker_thick = self.ui.spinBox_MarkerThickness.value()
        image_cctv, image_sat = displayOverlay_pts(self.image_cctv_color ,self.image_sat_color,self.K,self.R,self.T,self.disto,self.alpha, self.pts_cctv, self.pts_sat, self.marker_size, self.marker_thick, self.display_number, self.display_overlay, self.display_points)
        self.image_sat = image_sat
        self.image_cctv = image_cctv
        self.setSatImage()
        self.setCCTVImage()

    def getMarkerSize(self):
        self.marker_size = self.ui.spinBox_MarkerSize.value()
        image_cctv, image_sat = displayOverlay_pts(self.image_cctv_color ,self.image_sat_color,self.K,self.R,self.T,self.disto,self.alpha, self.pts_cctv, self.pts_sat, self.marker_size, self.marker_thick, self.display_number, self.display_overlay, self.display_points)
        self.image_sat = image_sat
        self.image_cctv = image_cctv
        self.setSatImage()
        self.setCCTVImage()
        
    ############### callback checkboxes #############
    def state_changed_DisplayPoints(self, int):
        if self.ui.checkBox_DisplayPoints.isChecked():
            self.display_points = True
        else:
            self.display_points = False
        image_cctv, image_sat = displayOverlay_pts(self.image_cctv_color ,self.image_sat_color,self.K,self.R,self.T,self.disto,self.alpha, self.pts_cctv, self.pts_sat, self.marker_size, self.marker_thick, self.display_number, self.display_overlay, self.display_points)
        self.image_sat = image_sat
        self.image_cctv = image_cctv
        self.setSatImage()
        self.setCCTVImage()

    def state_changed_DisplayOverlay(self, int):
        if self.ui.checkBox_DisplayOverlay.isChecked():
            self.display_overlay = True
        else:
            self.display_overlay = False
        image_cctv, image_sat = displayOverlay_pts(self.image_cctv_color ,self.image_sat_color,self.K,self.R,self.T,self.disto,self.alpha, self.pts_cctv, self.pts_sat, self.marker_size, self.marker_thick, self.display_number, self.display_overlay, self.display_points)
        self.image_sat = image_sat
        self.image_cctv = image_cctv
        self.setSatImage()
        self.setCCTVImage()

    def state_changed_DisplayPtsInd(self, int):
        if self.ui.checkBox_DisplayPtsInd.isChecked():
            self.display_number = True
        else:
            self.display_number = False
        image_cctv, image_sat = displayOverlay_pts(self.image_cctv_color ,self.image_sat_color,self.K,self.R,self.T,self.disto,self.alpha, self.pts_cctv, self.pts_sat, self.marker_size, self.marker_thick, self.display_number, self.display_overlay, self.display_points)
        self.image_sat = image_sat
        self.image_cctv = image_cctv
        self.setSatImage()
        self.setCCTVImage()


    ############### callback threshold #############

   
    def adjustAlphaSlide(self,value):
        self.alpha = float(value/100)
        self.ui.alpha_lineEdit.setText(str(self.alpha))
        image_cctv, image_sat = displayOverlay_pts(self.image_cctv_color ,self.image_sat_color,self.K,self.R,self.T,self.disto,self.alpha, self.pts_cctv, self.pts_sat, self.marker_size, self.marker_thick, self.display_number, self.display_overlay, self.display_points)
        #image_cctv, image_sat = displayOverlay(self.image_cctv_color ,self.image_sat_color,self.K,self.R,self.T,self.disto,self.alpha)
        self.image_sat = image_sat
        self.image_cctv = image_cctv
        self.setSatImage()
        self.setCCTVImage()


    def adjustAlphaEdit(self,text):
        self.alpha = float(text)
        self.ui.alpha_horizontalSlider.setValue(int(self.alpha*100))
        image_cctv, image_sat = displayOverlay_pts(self.image_cctv_color ,self.image_sat_color,self.K,self.R,self.T,self.disto,self.alpha, self.pts_cctv.astype(int), self.pts_sat.astype(int), self.marker_size, self.marker_thick, self.display_number, self.display_overlay, self.display_points)
        #image_cctv, image_sat = displayOverlay(self.image_cctv_color ,self.image_sat_color,self.K,self.R,self.T,self.disto,self.alpha)
        self.image_sat = image_sat
        self.image_cctv = image_cctv
        self.setSatImage()
        self.setCCTVImage()


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
