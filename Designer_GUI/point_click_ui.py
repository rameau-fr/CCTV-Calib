# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PointClick.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Keypoints_Dialog(object):
    def setupUi(self, Keypoints_Dialog):
        Keypoints_Dialog.setObjectName("Keypoints_Dialog")
        Keypoints_Dialog.resize(1055, 576)
        self.sat_im_graphicsView = QtWidgets.QGraphicsView(Keypoints_Dialog)
        self.sat_im_graphicsView.setGeometry(QtCore.QRect(10, 130, 491, 341))
        self.sat_im_graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.sat_im_graphicsView.setMouseTracking(False)
        self.sat_im_graphicsView.setFocusPolicy(QtCore.Qt.NoFocus)
        self.sat_im_graphicsView.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.sat_im_graphicsView.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.sat_im_graphicsView.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.sat_im_graphicsView.setObjectName("sat_im_graphicsView")
        self.sat_label = QtWidgets.QLabel(Keypoints_Dialog)
        self.sat_label.setGeometry(QtCore.QRect(10, 110, 102, 17))
        self.sat_label.setObjectName("sat_label")
        self.cctv_im_graphicsView = QtWidgets.QGraphicsView(Keypoints_Dialog)
        self.cctv_im_graphicsView.setGeometry(QtCore.QRect(530, 130, 491, 341))
        self.cctv_im_graphicsView.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.cctv_im_graphicsView.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.cctv_im_graphicsView.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.cctv_im_graphicsView.setObjectName("cctv_im_graphicsView")
        self.cctv_label = QtWidgets.QLabel(Keypoints_Dialog)
        self.cctv_label.setGeometry(QtCore.QRect(530, 110, 73, 17))
        self.cctv_label.setObjectName("cctv_label")
        self.layoutWidget = QtWidgets.QWidget(Keypoints_Dialog)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 20, 362, 28))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.NbPts_label = QtWidgets.QLabel(self.layoutWidget)
        self.NbPts_label.setObjectName("NbPts_label")
        self.horizontalLayout.addWidget(self.NbPts_label)
        self.NbPt_spinBox = QtWidgets.QSpinBox(self.layoutWidget)
        self.NbPt_spinBox.setMinimum(4)
        self.NbPt_spinBox.setProperty("value", 7)
        self.NbPt_spinBox.setObjectName("NbPt_spinBox")
        self.horizontalLayout.addWidget(self.NbPt_spinBox)
        self.checkBox = QtWidgets.QCheckBox(self.layoutWidget)
        self.checkBox.setChecked(True)
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout.addWidget(self.checkBox)
        self.layoutWidget1 = QtWidgets.QWidget(Keypoints_Dialog)
        self.layoutWidget1.setGeometry(QtCore.QRect(30, 60, 153, 28))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.CurrentPt_label = QtWidgets.QLabel(self.layoutWidget1)
        self.CurrentPt_label.setObjectName("CurrentPt_label")
        self.horizontalLayout_2.addWidget(self.CurrentPt_label)
        self.CurrentPt_spinBox = QtWidgets.QSpinBox(self.layoutWidget1)
        self.CurrentPt_spinBox.setObjectName("CurrentPt_spinBox")
        self.horizontalLayout_2.addWidget(self.CurrentPt_spinBox)
        self.layoutWidget2 = QtWidgets.QWidget(Keypoints_Dialog)
        self.layoutWidget2.setGeometry(QtCore.QRect(20, 520, 171, 28))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_10 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_4.addWidget(self.label_10)
        self.FontSizeSat_spinBox = QtWidgets.QSpinBox(self.layoutWidget2)
        self.FontSizeSat_spinBox.setMinimum(1)
        self.FontSizeSat_spinBox.setMaximum(200)
        self.FontSizeSat_spinBox.setProperty("value", 3)
        self.FontSizeSat_spinBox.setObjectName("FontSizeSat_spinBox")
        self.horizontalLayout_4.addWidget(self.FontSizeSat_spinBox)
        self.layoutWidget3 = QtWidgets.QWidget(Keypoints_Dialog)
        self.layoutWidget3.setGeometry(QtCore.QRect(530, 480, 457, 28))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.layoutWidget3)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_7 = QtWidgets.QLabel(self.layoutWidget3)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_5.addWidget(self.label_7)
        self.markerTypeCCTV_spinBox = QtWidgets.QSpinBox(self.layoutWidget3)
        self.markerTypeCCTV_spinBox.setMinimum(0)
        self.markerTypeCCTV_spinBox.setMaximum(6)
        self.markerTypeCCTV_spinBox.setProperty("value", 0)
        self.markerTypeCCTV_spinBox.setObjectName("markerTypeCCTV_spinBox")
        self.horizontalLayout_5.addWidget(self.markerTypeCCTV_spinBox)
        self.label_8 = QtWidgets.QLabel(self.layoutWidget3)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_5.addWidget(self.label_8)
        self.markerSizeCCTV_spinBox = QtWidgets.QSpinBox(self.layoutWidget3)
        self.markerSizeCCTV_spinBox.setMinimum(0)
        self.markerSizeCCTV_spinBox.setProperty("value", 12)
        self.markerSizeCCTV_spinBox.setObjectName("markerSizeCCTV_spinBox")
        self.horizontalLayout_5.addWidget(self.markerSizeCCTV_spinBox)
        self.label_9 = QtWidgets.QLabel(self.layoutWidget3)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_5.addWidget(self.label_9)
        self.markerThickCCTV_spinBox = QtWidgets.QSpinBox(self.layoutWidget3)
        self.markerThickCCTV_spinBox.setMinimum(1)
        self.markerThickCCTV_spinBox.setProperty("value", 3)
        self.markerThickCCTV_spinBox.setObjectName("markerThickCCTV_spinBox")
        self.horizontalLayout_5.addWidget(self.markerThickCCTV_spinBox)
        self.layoutWidget4 = QtWidgets.QWidget(Keypoints_Dialog)
        self.layoutWidget4.setGeometry(QtCore.QRect(20, 480, 457, 28))
        self.layoutWidget4.setObjectName("layoutWidget4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget4)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_4 = QtWidgets.QLabel(self.layoutWidget4)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_3.addWidget(self.label_4)
        self.markerTypeSat_spinBox = QtWidgets.QSpinBox(self.layoutWidget4)
        self.markerTypeSat_spinBox.setMinimum(0)
        self.markerTypeSat_spinBox.setMaximum(6)
        self.markerTypeSat_spinBox.setProperty("value", 0)
        self.markerTypeSat_spinBox.setObjectName("markerTypeSat_spinBox")
        self.horizontalLayout_3.addWidget(self.markerTypeSat_spinBox)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget4)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.markerSizeSat_spinBox = QtWidgets.QSpinBox(self.layoutWidget4)
        self.markerSizeSat_spinBox.setMinimum(0)
        self.markerSizeSat_spinBox.setProperty("value", 12)
        self.markerSizeSat_spinBox.setObjectName("markerSizeSat_spinBox")
        self.horizontalLayout_3.addWidget(self.markerSizeSat_spinBox)
        self.label_5 = QtWidgets.QLabel(self.layoutWidget4)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_3.addWidget(self.label_5)
        self.markerThickSat_spinBox = QtWidgets.QSpinBox(self.layoutWidget4)
        self.markerThickSat_spinBox.setMinimum(1)
        self.markerThickSat_spinBox.setProperty("value", 3)
        self.markerThickSat_spinBox.setObjectName("markerThickSat_spinBox")
        self.horizontalLayout_3.addWidget(self.markerThickSat_spinBox)
        self.layoutWidget5 = QtWidgets.QWidget(Keypoints_Dialog)
        self.layoutWidget5.setGeometry(QtCore.QRect(530, 520, 171, 28))
        self.layoutWidget5.setObjectName("layoutWidget5")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.layoutWidget5)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_11 = QtWidgets.QLabel(self.layoutWidget5)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_6.addWidget(self.label_11)
        self.FontSizeCCTV_spinBox = QtWidgets.QSpinBox(self.layoutWidget5)
        self.FontSizeCCTV_spinBox.setMinimum(1)
        self.FontSizeCCTV_spinBox.setMaximum(200)
        self.FontSizeCCTV_spinBox.setProperty("value", 3)
        self.FontSizeCCTV_spinBox.setObjectName("FontSizeCCTV_spinBox")
        self.horizontalLayout_6.addWidget(self.FontSizeCCTV_spinBox)
        self.layoutWidget6 = QtWidgets.QWidget(Keypoints_Dialog)
        self.layoutWidget6.setGeometry(QtCore.QRect(880, 20, 82, 58))
        self.layoutWidget6.setObjectName("layoutWidget6")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget6)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.help_pushButton = QtWidgets.QPushButton(self.layoutWidget6)
        self.help_pushButton.setObjectName("help_pushButton")
        self.verticalLayout.addWidget(self.help_pushButton)
        self.validate_buttonBox = QtWidgets.QDialogButtonBox(self.layoutWidget6)
        self.validate_buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Apply)
        self.validate_buttonBox.setObjectName("validate_buttonBox")
        self.verticalLayout.addWidget(self.validate_buttonBox)

        self.retranslateUi(Keypoints_Dialog)
        QtCore.QMetaObject.connectSlotsByName(Keypoints_Dialog)

    def retranslateUi(self, Keypoints_Dialog):
        _translate = QtCore.QCoreApplication.translate
        Keypoints_Dialog.setWindowTitle(_translate("Keypoints_Dialog", "keypoints selection"))
        self.sat_label.setText(_translate("Keypoints_Dialog", "Satellite image"))
        self.cctv_label.setText(_translate("Keypoints_Dialog", "cctv image"))
        self.NbPts_label.setText(_translate("Keypoints_Dialog", "Number of points:"))
        self.checkBox.setText(_translate("Keypoints_Dialog", "Display point\'s indexes"))
        self.CurrentPt_label.setText(_translate("Keypoints_Dialog", "current points:"))
        self.label_10.setText(_translate("Keypoints_Dialog", "Point index size:"))
        self.label_7.setText(_translate("Keypoints_Dialog", "marker type: "))
        self.label_8.setText(_translate("Keypoints_Dialog", "marker Size: "))
        self.label_9.setText(_translate("Keypoints_Dialog", "marker Thickness: "))
        self.label_4.setText(_translate("Keypoints_Dialog", "marker type: "))
        self.label_3.setText(_translate("Keypoints_Dialog", "marker Size: "))
        self.label_5.setText(_translate("Keypoints_Dialog", "marker Thickness: "))
        self.label_11.setText(_translate("Keypoints_Dialog", "Point index size:"))
        self.help_pushButton.setText(_translate("Keypoints_Dialog", "Help"))
