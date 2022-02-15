from Designer_GUI.point_click_ui import Ui_Keypoints_Dialog

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


palette = (2 ** 11 - 1, 2 ** 13 - 1, 2 ** 18 - 1)
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

# plot the points on the image
def plotPoints(image, points, marker_size, marker_thick, marker_type, display_index_flag, font_size ):
    image_disp = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    nb_pts = points.shape[0]
    for i in range(0,nb_pts):
        if(points[i,0]!=-1 and points[i,1]!=-1):
            #image_disp = cv2.circle(image, (points[i,0],points[i,1]), radius=pts_size, color=(0, 0, 255), thickness=1)
            color = compute_color_for_labels(i)
            image_disp = cv2.drawMarker(image, (points[i,0],points[i,1]),color, markerType=marker_type, markerSize=marker_size, thickness=marker_thick)
            if (display_index_flag==True):
                image_disp = cv2.putText(image, str(i), (points[i,0],points[i,1]), font, font_size,color, 3, cv2.LINE_AA)
    return image_disp



#Main window of the toolbox
class PointClickWindow(qtw.QWidget):

    closed = QtCore.pyqtSignal()

    def __init__(self, image_sat, image_cctv):
        super().__init__()
        self.ui = Ui_Keypoints_Dialog()
        self.ui.setupUi(self)

        #open the images
        self.image_sat = image_sat
        self.image_cctv = image_cctv
        self.image_sat_ori =  self.image_sat.copy()
        self.image_cctv_ori =  self.image_cctv.copy()
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
        #self.ui.sat_im_graphicsView.paintEvent = self.MypaintEvent
        
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

        #Points parameters
        self.number_points = 7
        self.ui.CurrentPt_spinBox.setMaximum(self.number_points-1)
        self.ui.NbPt_spinBox.valueChanged.connect(self.getNbOfPts)
        self.points_sat = np.full(( self.number_points,2), -1)
        self.points_cctv = np.full(( self.number_points,2), -1)
        self.marker_type_sat = self.ui.markerTypeSat_spinBox.value()
        self.marker_size_sat = self.ui.markerSizeSat_spinBox.value()
        self.marker_thick_sat = self.ui.markerThickSat_spinBox.value()
        self.marker_type_cctv = self.ui.markerTypeCCTV_spinBox.value()
        self.marker_size_cctv = self.ui.markerSizeCCTV_spinBox.value()
        self.marker_thick_cctv = self.ui.markerThickCCTV_spinBox.value()
        self.font_size_sat = self.ui.FontSizeSat_spinBox.value()
        self.font_size_cctv = self.ui.FontSizeCCTV_spinBox.value()
        self.ui.markerTypeSat_spinBox.valueChanged.connect(self.getMarkerTypeSat)
        self.ui.markerSizeSat_spinBox.valueChanged.connect(self.getMarkerSizeSat)
        self.ui.markerThickSat_spinBox.valueChanged.connect(self.getMarkerThickSat)
        self.ui.markerTypeCCTV_spinBox.valueChanged.connect(self.getMarkerTypeCCTV)
        self.ui.markerSizeCCTV_spinBox.valueChanged.connect(self.getMarkerSizeCCTV)
        self.ui.markerThickCCTV_spinBox.valueChanged.connect(self.getMarkerThickCCTV)
        self.ui.FontSizeSat_spinBox.valueChanged.connect(self.getFontSatSize)
        self.ui.FontSizeCCTV_spinBox.valueChanged.connect(self.getFontCCTVSize)

        #push button
        self.ui.help_pushButton.clicked.connect(self.helpdisplay)
        self.ui.validate_buttonBox.button(qtw.QDialogButtonBox.Apply).clicked.connect(self.validate_points)

        #validation flag
        self.validation = False
        
        
        

    def closeEvent(self, event):
        super(PointClickWindow, self).closeEvent(event)
        self.closed.emit()

    def validate_points(self):
        print(self.points_sat.shape[0])
        print(self.number_points)
        if ((np.sum(self.points_sat[:,1]>0) == self.number_points) and (np.sum(self.points_cctv[:,1]>0) == self.number_points)):
            print ("points selection done")
            self.validation = True
            self.close()

    def helpdisplay(self):
        QMessageBox.about(self, "Instructions", "Instructions: \n \
        - 1. Set the number of requiered points (at least 4) \n\
        - 2. Select a rough ROI in the Satellite image \n \
        - 3. Navigate the images and select one correspondance point on the satellite and CCTV images \n\
        - 4. Move to next point (Space of manual selection) \n \n \
        Navigate through the image:\n \
        - Zoom: mouse wheel \n\
        - Scroll: mouse wheel button press\n \
        - Select ROI: right click\n \
        - Next point and view reset to ROI: Space \n\
        - Next point: N") 

    ############### user config #############
    def getFontSatSize(self):
        self.font_size_sat = self.ui.FontSizeSat_spinBox.value()
        self.image_sat = plotPoints(self.image_sat_ori.copy(), self.points_sat, self.marker_size_sat, self.marker_thick_sat, self.marker_type_sat, self.ui.checkBox.isChecked(), self.font_size_sat)
        self.setSatImage()
    
    def getFontCCTVSize(self):
        self.font_size_cctv = self.ui.FontSizeCCTV_spinBox.value()
        self.image_cctv = plotPoints(self.image_cctv_ori.copy(), self.points_cctv, self.marker_size_cctv, self.marker_thick_cctv, self.marker_type_cctv, self.ui.checkBox.isChecked(), self.font_size_cctv)
        self.setCCTVImage()

    def getMarkerTypeSat(self):
        self.marker_type_sat = self.ui.markerTypeSat_spinBox.value()
        self.image_sat = plotPoints(self.image_sat_ori.copy(), self.points_sat, self.marker_size_sat, self.marker_thick_sat, self.marker_type_sat, self.ui.checkBox.isChecked(), self.font_size_sat)
        self.setSatImage()

    def getMarkerSizeSat(self):
        self.marker_size_sat = self.ui.markerSizeSat_spinBox.value()
        self.image_sat = plotPoints(self.image_sat_ori.copy(), self.points_sat, self.marker_size_sat, self.marker_thick_sat, self.marker_type_sat, self.ui.checkBox.isChecked(), self.font_size_sat)
        self.setSatImage()
    
    def getMarkerThickSat(self):
        self.marker_thick_sat = self.ui.markerThickSat_spinBox.value()
        self.image_sat = plotPoints(self.image_sat_ori.copy(), self.points_sat, self.marker_size_sat, self.marker_thick_sat, self.marker_type_sat, self.ui.checkBox.isChecked(), self.font_size_sat)
        self.setSatImage()

    def getMarkerTypeCCTV(self):
        self.marker_type_cctv = self.ui.markerTypeCCTV_spinBox.value()
        self.image_cctv = plotPoints(self.image_cctv_ori.copy(), self.points_cctv, self.marker_size_cctv, self.marker_thick_cctv, self.marker_type_cctv, self.ui.checkBox.isChecked(), self.font_size_cctv)
        self.setCCTVImage()

    def getMarkerSizeCCTV(self):
        self.marker_size_cctv = self.ui.markerSizeCCTV_spinBox.value()
        self.image_cctv = plotPoints(self.image_cctv_ori.copy(), self.points_cctv, self.marker_size_cctv, self.marker_thick_cctv, self.marker_type_cctv, self.ui.checkBox.isChecked(), self.font_size_cctv)
        self.setCCTVImage()
    
    def getMarkerThickCCTV(self):
        self.marker_thick_cctv = self.ui.markerThickCCTV_spinBox.value()
        self.image_cctv = plotPoints(self.image_cctv_ori.copy(), self.points_cctv, self.marker_size_cctv, self.marker_thick_cctv, self.marker_type_cctv, self.ui.checkBox.isChecked(), self.font_size_cctv)
        self.setCCTVImage()

    # method called by spin box
    def getNbOfPts(self):
        # getting current value
        self.number_points = self.ui.NbPt_spinBox.value()
        #update the maximum of the curent point selector
        self.ui.CurrentPt_spinBox.setMaximum(self.number_points-1)
        # update the points
        new_points_sat = np.full(( self.number_points,2), -1)
        new_points_cctv = np.full(( self.number_points,2), -1)
        prev_nb_pts = self.points_sat.shape[0]
        if(prev_nb_pts<=self.number_points):
            for i in range(0,prev_nb_pts):
                new_points_sat[i,:] = self.points_sat[i,:]
                new_points_cctv[i,:] = self.points_cctv[i,:]
        else:
            for i in range(0,self.number_points):
                new_points_sat[i,:] = self.points_sat[i,:]
                new_points_cctv[i,:] = self.points_cctv[i,:]
        self.points_sat = new_points_sat
        self.points_cctv = new_points_cctv
        print("new pts_sat", new_points_sat)
        
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            point_id = self.ui.CurrentPt_spinBox.value()
            if(point_id+1<self.number_points):
                self.ui.CurrentPt_spinBox.setValue(point_id+1)
            else:
                self.ui.CurrentPt_spinBox.setValue(0)
            self.resetViewToROISat()
            self.resetViewToROICCTV()

        elif event.key() == QtCore.Qt.Key_N:
            point_id = self.ui.CurrentPt_spinBox.value()
            if(point_id+1<self.number_points):
                self.ui.CurrentPt_spinBox.setValue(point_id+1)
            else:
                self.ui.CurrentPt_spinBox.setValue(0)
        event.accept()
       
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
        # Left-click to select a point
        if event.button() == QtCore.Qt.LeftButton:
            Loc =  self.ui.sat_im_graphicsView.mapToScene(event.pos())
            point_id = self.ui.CurrentPt_spinBox.value()
            self.points_sat[point_id,0] = Loc.x()
            self.points_sat[point_id,1] = Loc.y()
            #update the plot
            self.image_sat = plotPoints(self.image_sat_ori.copy(), self.points_sat, self.marker_size_sat, self.marker_thick_sat, self.marker_type_sat, self.ui.checkBox.isChecked(), self.font_size_sat)
            self.setSatImage()
            print("points sat :: ",self.points_sat)
            print("points cctv :: ",self.points_cctv)

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
        # Left-click to select a point
        if event.button() == QtCore.Qt.LeftButton:
            Loc =  self.ui.cctv_im_graphicsView.mapToScene(event.pos())
            point_id = self.ui.CurrentPt_spinBox.value()
            self.points_cctv[point_id,0] = Loc.x()
            self.points_cctv[point_id,1] = Loc.y()
            #update the plot
            self.image_cctv = plotPoints(self.image_cctv_ori.copy(), self.points_cctv, self.marker_size_cctv, self.marker_thick_cctv, self.marker_type_cctv, self.ui.checkBox.isChecked(), self.font_size_cctv)
            self.setCCTVImage()
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

