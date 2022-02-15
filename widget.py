# This Python file uses the following encoding: utf-8
import os
from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QComboBox, QSpinBox, QFileDialog, QCheckBox, QVBoxLayout
from PySide6.QtCore import QFile, QTimer, QDir
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPixmap, QImage, qRgb

import AERGen as ag
import configmanager as cm
import main
import numpy as np


class Widget(QWidget):
    def __init__(self):
        super(Widget, self).__init__()
        self.load_ui()

        self.newDatasetInputs = {
        "initialPosition": {"x":self.findChild(QSpinBox, "x0") , "y":self.findChild(QSpinBox, "y0")},
        "endPosition": {"x":self.findChild(QSpinBox, "x1") , "y":self.findChild(QSpinBox, "y1")},
        "radius":self.findChild(QSpinBox, "radius"),
        "speed":self.findChild(QSpinBox, "speed")
        }

        self.previewNew = self.findChild(QPushButton, "previewNew")
        self.previewNew.clicked.connect(self._previewNew)
        self.appendToExisting = self.findChild(QPushButton, "appendToExisting")
        self.appendToExisting.clicked.connect(self._appendToExisting)
        self.saveAsNew = self.findChild(QPushButton, "saveAsNew")
        self.saveAsNew.clicked.connect(self._saveAsNew)

        self.datasetCombo = self.findChild(QComboBox, "datasetCombo")
        self.datasetLoad = self.findChild(QPushButton, "datasetLoad")
        self.datasetLoad.clicked.connect(self._datasetLoad)
        self.datasetPreview = self.findChild(QPushButton, "datasetPreview")
        self.datasetPreview.clicked.connect(self._datasetPreview)
        self.datasetPreview.setEnabled(False)

        self.chooseConf = self.findChild(QPushButton, "chooseConf")
        self.chooseConf.clicked.connect(self._chooseConf)
        self.numOfSets = self.findChild(QComboBox, "numOfSets")
        self.letsGo = self.findChild(QPushButton, "letsGo")
        self.letsGo.clicked.connect(self._letsGo)

        self.preview = self.findChild(QLabel, "preview")
        self.runLearning = self.findChild(QLabel, "trainResult")
        self.listOfSets = self.findChild(QVBoxLayout, "listOfSets")
        self.chosenSets = []

        self.dataset = None
        self.network_conf = ""

        self.renderComplete = False
        self.renderTimer = None

        self.db = cm.DBManager()

        self.loadTraces()

        self.path = QDir(os.fspath(Path(__file__).resolve().parent))

    def load_ui(self):
        loader = QUiLoader()
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        loader.load(ui_file, self)
        ui_file.close()

    def loadTraces(self):
        self.datasetCombo.clear()
        traces = self.db.read_trace_entries()
        for trace in traces:
            self.datasetCombo.addItem(trace.trace_alias, userData=trace.trace_path)
            button = QCheckBox(trace.trace_alias)
            button.stateChanged.connect(self.toggleDataSet(trace.trace_path))
            self.listOfSets.insertWidget(0, button)

    def toggleDataSet(self, alias):
        def _(state):
            if not state:
                self.chosenSets.remove(alias)
            else:
                self.chosenSets.append(alias)
        return _

    def _datasetPreview(self):
        self.renderPreview()

    def _previewNew(self):
        self.generateDataset()
        self.renderPreview()

    def generateDataset(self, time = 0):
        self.dataset = []
        raw_dataset = ag.AERGen(radius=self.newDatasetInputs["radius"].value(),
                                speed=self.newDatasetInputs["speed"].value(),
                                pos_start = ag.Position(self.newDatasetInputs["initialPosition"]["x"].value(),
                                                        self.newDatasetInputs["initialPosition"]["y"].value()),
                                pos_end = ag.Position(self.newDatasetInputs["endPosition"]["x"].value(),
                                                      self.newDatasetInputs["endPosition"]["y"].value()),
                                start_time = time)
        for events in raw_dataset:
            for ev in events:
                self.dataset.append(ev)


    def renderPreview(self):
        self.renderTimer = QTimer(self)
        self.renderTimer.timeout.connect(self.showNextFrame)
        self.renderTimer.timeout.connect(self.update)
        self.genNextFrame = self._genNextFrame()
        self.renderComplete = False
        self.showNextFrame()

    def showNextFrame(self):
        if not self.renderComplete:
            try:
                self.preview.setPixmap(next(self.genNextFrame))
            except StopIteration:
                self.renderComplete = True
            self.renderTimer.start(10)

    def _genNextFrame(self):
        white_square = np.ones((28,28), dtype=np.uint8)
        current_frame = np.copy(white_square)
        frames_shown = 0
        colortable = [qRgb(i, i, i) for i in range(256)]
        for ev in self.dataset:
            x_coord = ev.position.x
            y_coord = ev.position.y
            color = ev.polarity * 255
            try:
                current_frame[y_coord-1][x_coord-1] = color
            except:
                pass
            frames_shown += 1
            img = QImage(current_frame, 28, 28, 28, QImage.Format_Indexed8)
            img.setColorTable(colortable)
            pixmap = QPixmap(img).scaledToHeight(300)
            yield pixmap

    def _appendToExisting(self):
        filename = self.path.relativeFilePath(QFileDialog.getOpenFileName(self, "Open file", "")[0])
        self._datasetLoad(path = filename)
        self.generateDataset(time = self.dataset[-1].time)
        with open(filename[0], 'a') as f:
            f.write(' '.join([ag.aer_encode(ev) for ev in self.dataset]))
        trace_entry = self.db.get_trace_by_path(filename)
        new_entry = cm.Trace(trace_entry.id, trace_entry.trace_path, trace_entry.trace_alias, trace_entry.speed, self.dataset[-1].time, "")
        self.db.update(new_entry)

    def _saveAsNew(self):
        filename = QFileDialog.getSaveFileName(self, "Save file", "", ".bin")
        self.generateDataset()
        with open(''.join(filename), 'w') as f:
            f.write(' '.join([ag.aer_encode(ev)[0] for ev in self.dataset]))
            self.db.add_trace_entry(filename[0].split('/')[-1], trace_path=self.path.relativeFilePath(''.join(filename)), target_speed=self.newDatasetInputs["speed"].value(), end_time=self.dataset[-1].time)
        self.loadTraces()

    def _datasetLoad(self, path = ""):
        if not path:
            path = self.datasetCombo.currentData()
        with open(path, 'r') as f:
            self.dataset = [ag.aer_decode(ev) for ev in f.readline().split(' ')]
        self.datasetPreview.setEnabled(True)

    def _chooseConf(self):
        self.network_conf = self.path.relativeFilePath(QFileDialog.getOpenFileName(self, "Open file", "")[0])

    def _letsGo(self):
        if not self.network_conf:
            return
        model, feed = main.construct_network("iter", self.network_conf)
        
        datasets = []
        for path in self.chosenSets:
            with open(path, 'r') as f:
                datasets.append([ag.aer_decode(ev) for ev in f.readline().split(' ')])

if __name__ == "__main__":
    app = QApplication([])
    widget = Widget()
    widget.show()
    sys.exit(app.exec_())
