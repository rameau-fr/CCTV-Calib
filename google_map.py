from Designer_GUI.google_map_ui import Ui_googlemap_Dialog

from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QPixmap, QImage   
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtWidgets import QMessageBox

import numpy as np
import random
import math

import requests

import cv2

import pdb

#generate the sat image
def write_sat_image_google_map(API_KEY, ZOOM, w_res, h_res, scale, longi, lati):
    # updating the URL
    BASE_URL = "https://maps.googleapis.com/maps/api/staticmap?"
    center = str(longi) + "," + str(lati)
    URL = BASE_URL + "center=" + center + "&zoom=" + str(ZOOM) + "&size=" + str(w_res) + "x" + str(h_res) + "&scale=" + str(scale) + "&maptype=satellite" + "&format=png" + "&key=" + API_KEY
    
    # HTTP request
    response = requests.get(URL)
    # storing the response in a file (image)
    with open('temp_sat.png', 'wb') as file:
        # writing data into the file
        file.write(response.content)

#from https://stackoverflow.com/questions/47106276/converting-pixels-to-latlng-coordinates-from-google-static-image
def getPointLatLng(x, y, zoom, lat, lng,h,w):
    parallelMultiplier = math.cos(lat * math.pi / 180)
    degreesPerPixelX = 360 / math.pow(2, zoom + 8)
    degreesPerPixelY = 360 / math.pow(2, zoom + 8) * parallelMultiplier
    pointLat = lat - degreesPerPixelY * ( y - h / 2)
    pointLng = lng + degreesPerPixelX * ( x  - w / 2)

    return (pointLat, pointLng)


#Main window of the toolbox
class GoogleMapWindow(qtw.QWidget):

    closed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.ui = Ui_googlemap_Dialog()
        self.ui.setupUi(self)

        #parameters
        self.longi = self.ui.lon_lineEdit.text()
        self.lati = self.ui.lat_lineEdit.text()
        self.zoom = self.ui.zoom_lineEdit.text()
        self.width = self.ui.width_lineEdit.text()
        self.height = self.ui.height_lineEdit.text()
        self.scale = self.ui.scale_lineEdit.text()
        self.API = self.ui.APIKey_lineEdit.text() 

        #keypoints
        self.pts_gps = []
        self.pts_im = []

        #parameters zoom and drag
        self.ui.graphicsView.wheelEvent = self.scaleSceneSat
        self.ui.graphicsView.setDragMode(qtw.QGraphicsView.ScrollHandDrag)
        self.ui.graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self._mousePressed = None
        self.ui.graphicsView.mousePressEvent = self.MymousePressEventSat
        self.ui.graphicsView.mouseMoveEvent = self.MymouseMoveEventSat
        self.ui.graphicsView.mouseReleaseEvent = self.MymouseReleaseEventSat

        #image
        self.image = np.array([[0,0],[0,0]])

        #buttons
        self.ui.generate_sat_pushButton.clicked.connect(self.press_generate)
        self.ui.save_pushButton.clicked.connect(self.press_save)

    def press_save(self):
        test = 1
        if len(self.pts_im)>0:
            name = qtw.QFileDialog.getSaveFileName(self, 'Save File','./',filter='image (*.png)')
            save_path = name[0]
            file_name = name[0].rsplit('/', 1)[-1]
            save_path_image = name[0] + '.png'
            save_path_point = name[0] + '.yml'
            print(save_path_image)
            print(save_path_point)

            #save image
            cv2.imwrite(save_path_image,self.image)

            #save keypoints
            cv_file = cv2.FileStorage(save_path_point, cv2.FILE_STORAGE_WRITE)
            cv_file.write("pts_gps", self.pts_gps)
            cv_file.write("pts_sat", self.pts_im)
            cv_file.release()
        


    def press_generate(self):
        self.longi = self.ui.lon_lineEdit.text()
        self.lati = self.ui.lat_lineEdit.text()
        self.zoom = self.ui.zoom_lineEdit.text()
        self.width = self.ui.width_lineEdit.text()
        self.height = self.ui.height_lineEdit.text()
        self.scale = self.ui.scale_lineEdit.text()
        self.API = self.ui.APIKey_lineEdit.text() 

        #generate the image
        write_sat_image_google_map(self.API, self.zoom, self.width, self.height, self.scale, self.longi, self.lati)

        #Read the image
        self.image = cv2.imread("temp_sat.png")

        #display the image
        self.setSatImage()

        #compute the 4 corner point gps locations 
        Top_left_gps = np.asarray(getPointLatLng(0, 0, float(self.zoom), float(self.longi), float(self.lati), float(self.height), float(self.width)))
        Top_right_gps = np.asarray(getPointLatLng(float(self.width), 0, float(self.zoom), float(self.longi), float(self.lati), float(self.height), float(self.width)))
        bottom_right_gps = np.asarray(getPointLatLng(float(self.width), float(self.height), float(self.zoom), float(self.longi), float(self.lati), float(self.height), float(self.width)))
        bottom_left_gps = np.asarray(getPointLatLng(0, float(self.height), float(self.zoom), float(self.longi), float(self.lati), float(self.height), float(self.width)))
        self.pts_gps = np.vstack((Top_left_gps,Top_right_gps,bottom_right_gps,bottom_left_gps))
        self.pts_im = np.array([[0,0],[float(self.width)*float(self.scale), 0],[float(self.width)*float(self.scale), float(self.height)*float(self.scale)],[0, float(self.height)*float(self.scale)]])

    def closeEvent(self, event):
        super(GoogleMapWindow, self).closeEvent(event)
        self.closed.emit()

    def MymousePressEventSat(self, event):
         # Left-click-to-pan
        if event.button() == QtCore.Qt.LeftButton:
            self._mousePressed = QtCore.Qt.LeftButton
            self._mousePressedPos = event.pos()
            self.ui.graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.ClosedHandCursor))
            self._dragPos = event.pos()     

    def MymouseMoveEventSat(self, event):
        mouse_position = self.ui.graphicsView.mapToScene(event.pos())
        self.mouse_position = [mouse_position.x(), mouse_position.y()]
        # Left-click-to-pan
        if self._mousePressed == QtCore.Qt.LeftButton:
            newPos = event.pos()
            diff = newPos - self._dragPos
            self._dragPos = newPos
            self.ui.graphicsView.horizontalScrollBar().setValue(self.ui.graphicsView.horizontalScrollBar().value() - diff.x())
            self.ui.graphicsView.verticalScrollBar().setValue(self.ui.graphicsView.verticalScrollBar().value() - diff.y())
            
    def MymouseReleaseEventSat(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.ui.graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.ArrowCursor))
            self._mousePressed = None
            self.ui.graphicsView.update()
            
    def scaleSceneSat(self, event):
        self.ui.graphicsView.setTransformationAnchor(qtw.QGraphicsView.NoAnchor)
        self.ui.graphicsView.setResizeAnchor(qtw.QGraphicsView.NoAnchor)
        oldPos = self.ui.graphicsView.mapToScene(event.pos())
        delta = 1.0015**event.angleDelta().y()
        self.ui.graphicsView.scale(delta, delta)
        newPos = self.ui.graphicsView.mapToScene(event.pos())
        delta = newPos - oldPos
        self.ui.graphicsView.translate(delta.x(), delta.y())
        self.ui.graphicsView.update()
    
    def resetViewToROISat(self):
        self.ui.graphicsView.fitInView(self.sat_roi, QtCore.Qt.KeepAspectRatio)
        self.ui.graphicsView.update()

    def updateView(self):
        scene_sat = self.ui.graphicsView.scene()
        scene_cctv = self.ui.graphicsView.scene()
        if scene_sat is not None:
            r_sat = scene_sat.sceneRect()
            self.ui.graphicsView.fitInView(r_sat, QtCore.Qt.KeepAspectRatio)
        if scene_cctv is not None:    
            r_cctv = scene_cctv.sceneRect()
            self.ui.graphicsView.fitInView(r_cctv, QtCore.Qt.KeepAspectRatio)
        
    def resizeEvent(self, event):
        self.updateView()

    def showEvent(self, event):
        if not event.spontaneous():
            self.updateView()

    def setSatImage(self):
        """
        display the satellite image
        """
        image = self.image
        self.image_sat_disp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(self.image_sat_disp, self.image_sat_disp.shape[1],self.image_sat_disp.shape[0], self.image_sat_disp.strides[0],QImage.Format_RGB888)
        scene = qtw.QGraphicsScene(self)
        pixmap = QPixmap.fromImage(image)
        item = qtw.QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.ui.graphicsView.setScene(scene)
        self._start_sat = QtCore.QPointF()
        self._current_rect_item_sat = qtw.QGraphicsRectItem()
        self._current_rect_item_sat.setBrush(QtGui.QColor(255,50,50, 60))
        self._current_rect_item_sat.setFlag(qtw.QGraphicsItem.ItemIsMovable, True)
        self.ui.graphicsView.scene().addItem(self._current_rect_item_sat)
        #self.sat_roi = self.ui.sat_image_graphicsView.scene().sceneRect() 
        self.ui.graphicsView.update()
   
if __name__ == "__main__":

    app = qtw.QApplication([])
    widget = GoogleMapWindow()
    widget.show()
    app.exec_()
