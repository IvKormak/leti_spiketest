# This Python file uses the following encoding: utf-8
import os
import pickle
import random
import sys
import time
from pathlib import Path
from concurrent import futures
import numpy as np

from PySide6.QtCore import QFile, QTimer, QDir, Qt, QObject, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, qRgb
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QComboBox, QSpinBox, QFileDialog, QCheckBox, \
    QVBoxLayout, QScrollArea, QSpacerItem, QHBoxLayout

import AERGen as ag
import configmanager as cm
import main

from checkablecombobox import CheckableComboBox as CComboBox


class MainWindow(QWidget):
    start_training = Signal(list)
    finish_training = Signal()
    start_reckon = Signal()
    finish_reckon = Signal()

    def __init__(self):
        super(MainWindow, self).__init__()
        self.load_ui()

        self.newDatasetInputs = {
            "initialPosition": {"x": self.findChild(QSpinBox, "x0"), "y": self.findChild(QSpinBox, "y0")},
            "endPosition": {"x": self.findChild(QSpinBox, "x1"), "y": self.findChild(QSpinBox, "y1")},
            "radius": self.findChild(QSpinBox, "radius"),
            "speed": self.findChild(QSpinBox, "speed")
        }

        self.path = QDir(os.fspath(Path(__file__).resolve().parent))

        self.chosenSets = []
        self.chosenReckonSets = []
        self.chosenToRemoveSets = []

        self.db = cm.DBManager()

        self.statusbar = self.findChild(QLabel, "statusbar")

        self.previewNew = self.findChild(QPushButton, "previewNew")
        self.appendToExisting = self.findChild(QPushButton, "appendToExisting")
        self.saveAsNew = self.findChild(QPushButton, "saveAsNew")

        self.datasetCombo = self.findChild(QComboBox, "datasetCombo")
        self.datasetLoad = self.findChild(QPushButton, "datasetLoad")
        self.datasetPreview = self.findChild(QPushButton, "datasetPreview")
        self.datasetPreview.setEnabled(False)
        self.datasetDelete = self.findChild(QPushButton, "datasetDelete")

        self.chooseConf = self.findChild(QPushButton, "chooseConf")
        self.numOfSets = self.findChild(QSpinBox, "numOfSets")
        self.poolSize = self.findChild(QSpinBox, "poolSize")
        self._poolSize = 1
        self.donorModels = []
        self.letsGo = self.findChild(QPushButton, "letsGo")
        self.trainResult = self.findChild(QLabel, "trainResult")

        self.preview = self.findChild(QLabel, "preview")
        self.runLearning = self.findChild(QLabel, "trainResult")
        self.setsContainer = self.findChild(QVBoxLayout, "setsContainer")
        self.setsLayout = QVBoxLayout()

        self.hl5 = self.findChild(QHBoxLayout, "horizontalLayout_5")
        self.datasetCombo_2 = CComboBox()
        self.hl5.insertWidget(0, self.datasetCombo_2)

        self.chooseModel = self.findChild(QPushButton, "chooseModel")
        self.deleteSets = self.findChild(QPushButton, "deleteSets")
        self.reckonSetsLayout = QVBoxLayout()
        self.reckonSetsContainer = self.findChild(QVBoxLayout, "reckonSetsContainer")
        self.reckonSetsInnerContainer = QScrollArea()
        self.reckonSetsContainer.insertWidget(0, self.reckonSetsInnerContainer)
        self.addSet = self.findChild(QPushButton, "addSet")
        self.answers = self.findChild(QLabel, "answers")
        self.reckonGraphics = self.findChild(QLabel, "reckonGraphics")
        self.reckon = self.findChild(QPushButton, "reckon")
        self.usePool = self.findChild(QCheckBox, "usePool")
        self.reckon.setEnabled(False)
        self.reckonPixmap = None

        self.previewNew.clicked.connect(self.previewNew_handler)
        self.appendToExisting.clicked.connect(self.appendToExisting_handler)
        self.saveAsNew.clicked.connect(self.saveAsNew_handler)
        self.datasetLoad.clicked.connect(self.datasetLoad_handler)
        self.datasetPreview.clicked.connect(self.datasetPreview_handler)
        self.datasetDelete.clicked.connect(self.datasetDelete_handler)
        self.chooseConf.clicked.connect(self.chooseConf_handler)
        self.letsGo.clicked.connect(self.letsGo_handler)
        self.chooseModel.clicked.connect(self.chooseModel_handler)
        self.deleteSets.clicked.connect(self.deleteSets_handler)
        self.addSet.clicked.connect(self.addSet_handler)
        self.reckon.clicked.connect(self.startReckon_handler)


        widget = QWidget()
        widget.setLayout(self.setsLayout)

        scroll = QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)

        self.setsContainer.insertWidget(0, scroll)

        self.load_traces()

        self.dataset = None
        self.network_conf = ""

        self.renderComplete = False
        self.renderTimer = None

        self.colortable = [qRgb(i, i, i) for i in range(256)]

        self.trainer = None
        self.trainer_thread = None


    def load_ui(self):
        loader = QUiLoader()
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        loader.load(ui_file, self)
        ui_file.close()

    def set_status(self, text):
        self.statusbar.setText(text)
        self.repaint()

    def load_traces(self):
        self.set_status('Начинаем загрузку')
        self.datasetCombo.clear()
        self.datasetCombo_2.clear()
        traces = self.db.read_trace_entries()

        child = self.setsLayout.takeAt(0)
        while child != None:
            child = self.setsLayout.takeAt(0)
            del child.widget
            del child

        for trace in traces:
            self.datasetCombo.addItem(trace.trace_alias, userData=trace.trace_path)
            self.datasetCombo_2.addItem(trace.trace_alias, userData=trace.trace_path)
            button = QCheckBox(trace.trace_alias)
            button.stateChanged.connect(self.toggle_dataset(trace.trace_path, self.chosenSets))
            self.setsLayout.insertWidget(0, button)
        self.setsLayout.addStretch()
        self.set_status('Готово')

    def addSet_handler(self):
        for path in self.datasetCombo_2.currentData():
            self.chosenReckonSets.append(self.db.get_trace_by_path(path))
        self.render_reckon_sets()

    def deleteSets_handler(self):
        self.chosenReckonSets = [i for i in self.chosenReckonSets if i.trace_path not in self.chosenToRemoveSets]
        self.render_reckon_sets()

    def chooseModel_handler(self):
        p = QFileDialog.getOpenFileName(self, "Выберите файл с обученной моделью", filter="*.pkl")
        if any(p):
            with open(self.path.relativeFilePath(p[0]), "rb") as f:
                self.reckonModel = pickle.load(f)
        self.reckon.setEnabled(True)

    def fill_labels(self, labels:dict):
        s = ""
        for k in labels.keys():
            s += f"{k}\n"
        self.answers.setText(s)

    def render_reckon_image(self, events, shape):
        if self.reckonPixmap is None:
            white_square = np.ones(shape, dtype=np.uint8)
            self.reckonPixmap = np.copy(white_square)
        for ev in events:
            x_coord = ev.position.x
            y_coord = ev.position.y
            color = ev.polarity * 255
            if 0 <= x_coord < shape[0] and 0 <= y_coord < shape[1]:
                self.reckonPixmap[y_coord][x_coord] = color
            img = QImage(self.reckonPixmap, shape[0], shape[1], shape[0], QImage.Format_Indexed8)
            img.setColorTable(self.colortable)
            pixmap = QPixmap(img).scaledToHeight(300)
            self.reckonGraphics.setPixmap(pixmap)

    def render_reckon_sets(self):
        self.reckonSetsContainer.removeWidget(self.reckonSetsInnerContainer)
        self.reckonSetsLayout = QVBoxLayout()

        for trace in self.chosenReckonSets:
            button = QCheckBox(trace.trace_alias)
            button.stateChanged.connect(self.toggle_dataset(trace.trace_path, self.chosenToRemoveSets))
            self.reckonSetsLayout.insertWidget(0, button)
        self.reckonSetsLayout.addStretch()

        widget = QWidget()
        widget.setLayout(self.reckonSetsLayout)

        self.reckonSetsInnerContainer = QScrollArea()
        self.reckonSetsInnerContainer.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.reckonSetsInnerContainer.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.reckonSetsInnerContainer.setWidgetResizable(True)
        self.reckonSetsInnerContainer.setWidget(widget)

        self.reckonSetsContainer.insertWidget(0, self.reckonSetsInnerContainer)

    def startReckon_handler(self):
        self.reckonthread = QThread()
        self.reckonthread.start()
        self.reckognizer = ReckognizerWorker(self.reckonModel, self.chosenReckonSets)
        self.reckognizer.moveToThread(self.reckonthread)
        self.reckognizer.done.connect(self.end_reckon)
        self.reckognizer.events.connect(self.render_reckon_image)
        self.reckognizer.events.connect(self.update)
        self.reckognizer.reckon.connect(self.fillLabels)
        self.end_reckon.connect(self.reckonthread.quit)
        self.start_reckon.connect(self.reckognizer.reckognize_frame)
        self.start_reckon.emit()

    def end_reckon(self):
        self.reckonPixmap = None
        self.end_reckon.emit()
        del self.reckognizer
        del self.reckonthread

    def datasetDelete_handler(self):
        self.db.delete_trace_entry(self.datasetCombo.currentData())
        self.load_traces()

    def toggle_dataset(self, alias, variable):
        def toggle(state):
            if not state:
               variable.remove(alias)
            else:
                variable.append(alias)

        return toggle

    def datasetPreview_handler(self):
        self.render_preview()

    def previewNew_handler(self):
        self.generate_dataset()
        self.render_preview()

    def generate_dataset(self, time=0):
        self.set_status('Начинаем генерацию траектории')
        self.dataset = []
        raw_dataset = ag.AERGen(radius=self.newDatasetInputs["radius"].value(),
                                 speed=self.newDatasetInputs["speed"].value(),
                                 pos_start=ag.Position(self.newDatasetInputs["initialPosition"]["x"].value(),
                                                       self.newDatasetInputs["initialPosition"]["y"].value()),
                                 pos_end=ag.Position(self.newDatasetInputs["endPosition"]["x"].value(),
                                                     self.newDatasetInputs["endPosition"]["y"].value()),
                                 start_time=time)
        for ev in raw_dataset:
            self.dataset += ev

    def render_preview(self):
        self.set_status('Начинаем показ траектории')
        self.renderTimer = QTimer(self)
        self.iterNextFrame = self._gen_next_frame((28, 28))
        self.renderTimer.timeout.connect(self.show_next_frame)
        self.renderTimer.timeout.connect(self.update)
        self.renderComplete = False
        self.show_next_frame()

    def show_next_frame(self):
        if not self.renderComplete:
            try:
                self.preview.setPixmap(next(self.iterNextFrame))
            except StopIteration:
                self.renderComplete = True
                self.set_status('Готово')
            self.renderTimer.start(10)

    def _gen_next_frame(self, shape):
        white_square = np.ones(shape, dtype=np.uint8)
        current_frame = np.copy(white_square)
        frames_shown = 0
        for ev in self.dataset:
            x_coord = ev.position.x
            y_coord = ev.position.y
            color = ev.polarity * 255
            if 0 <= x_coord < shape[0] and 0 <= y_coord < shape[1]:
                current_frame[y_coord][x_coord] = color
            frames_shown += 1
            img = QImage(current_frame, shape[0], shape[1], shape[0], QImage.Format_Indexed8)
            img.setColorTable(self.colortable)
            pixmap = QPixmap(img).scaledToHeight(300)
            yield pixmap

    def appendToExisting_handler(self):
        filename = self.path.relativeFilePath(QFileDialog.getOpenFileName(self, "Open file", "")[0])
        self.datasetLoad_handler(path=filename)
        self.generate_dataset(time=self.dataset[-1].time)
        with open(filename[0], 'a') as f:
            f.write(' '.join([ag.aer_encode(ev) for ev in self.dataset]))
        trace_entry = self.db.get_trace_by_path(filename)
        new_entry = cm.Trace(trace_entry.id, trace_entry.trace_path, trace_entry.trace_alias, trace_entry.speed,
                             self.dataset[-1].time, "")
        self.db.update(new_entry)
        self.set_status(f'Траектория добавлена к файлу {filename}')

    def saveAsNew_handler(self):
        filename = QFileDialog.getSaveFileName(self, "Save file", "", ".bin")
        self.generate_dataset()
        with open(''.join(filename), 'w') as f:
            f.write(' '.join([ag.aer_encode(ev)[0] for ev in self.dataset]))
            self.db.add_trace_entry(filename[0].split('/')[-1],
                                    trace_path=self.path.relativeFilePath(''.join(filename)),
                                    target_speed=self.newDatasetInputs["speed"].value(), end_time=self.dataset[-1].time)
        self.load_traces()
        self.set_status(f'Сохранено в файл {filename}')

    def datasetLoad_handler(self, path=""):
        if not path:
            path = self.datasetCombo.currentData()
        with open(path, 'r') as f:
            self.dataset = [ag.aer_decode(ev) for ev in f.readline().split(' ')]
        self.datasetPreview.setEnabled(True)
        self.set_status(f'Загружен файл {path}')

    def chooseConf_handler(self):
        self.network_conf = self.path.relativeFilePath(
            QFileDialog.getOpenFileName(self, "Выберите файл с конфигурацией модели", "")[0])
        self.letsGo.setEnabled(True)

    def letsGo_handler(self):
        self.timestart = time.time()
        if not self.network_conf:
            self.set_status("Конфигурация не выбрана!")
            return
        self.letsGo.setEnabled(False)
        self._poolSize = self.poolSize.value() if self.usePool.checkState() else 1
        self.tracestogo = self._poolSize*self.numOfSets.value()
        self.tracesdone = 0
        neuron_changes = {}
        general_changes = {"pool_size": str(self._poolSize), "epoch_length": str(self.numOfSets.value())}
        self.target_model, feed = main.construct_network(feed_type="iter",
                                                         source_file=self.network_conf,
                                                         learn=False,
                                                         update_neuron_parameters=neuron_changes,
                                                         update_general_parameters=general_changes)


        self.trainer_thread = QThread()
        self.trainer_thread.start()
        self.trainer = TrainerWorker(pool_size=self._poolSize,
                                     feed_type="iter",
                                     source_file=self.network_conf,
                                     learn=True,
                                     update_neuron_parameters=neuron_changes,
                                     update_general_parameters=general_changes)
        self.trainer.moveToThread(self.trainer_thread)
        self.start_training.connect(self.trainer.train)
        self.trainer.done.connect(self.training_finished)
        self.trainer.trace_finished.connect(self.count_progress)
        self.finish_training.connect(self.trainer_thread.quit)
        self.start_training.emit(self.chosenSets)
        self.set_status('Начинаем обучение')

    def count_progress(self):
        self.tracesdone += 1
        self.set_status(f"Завершено {np.around(self.tracesdone / self.tracestogo * 100, decimals=2)}%")

    def training_finished(self, donor_model):
        self.donorModels.append(donor_model)
        self._poolSize -= 1
        self.set_status(f"Обучение сети {self.poolSize.value() - self._poolSize} из {self.poolSize.value()} закончено")
        if not self._poolSize:
            if self.usePool.checkState():
                labels = main.fill_model_from_pool(self.target_model, self.donorModels)
                if len(labels) == len(self.chosenSets):
                    self.finish_training.emit()
                    self.set_status(f"Времени затрачено: {time.time() - self.timestart}")
                    folderToSave = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения файлов", "")
                    main.save_attention_maps(self.target_model, folderToSave)

                    with open(f"{folderToSave}/model.pkl", "wb") as f:
                        pickle.dump(self.target_model, f)

                    main.glue_attention_maps(self.target_model, folderToSave)
                    img = QImage(f"{folderToSave}/attention_maps.png")
                    pixmap = QPixmap(img).scaledToHeight(300)

                    self.trainResult.setPixmap(pixmap)
                    self.letsGo.setEnabled(True)

                    del self.trainer_thread
                    del self.trainer
                else:
                    self._poolSize = self.poolSize.value()
                    self.start_training.emit(self.chosenSets)
                    self.set_status('Начинаем обучение заново')
            else:
                self.finish_training.emit()
                self.set_status(f"Времени затрачено: {time.time() - self.timestart}")
                folderToSave = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения файлов", "")
                main.save_attention_maps(self.donorModels[0], folderToSave)

                with open(f"{folderToSave}/model.pkl", "wb") as f:
                    pickle.dump(self.donorModels[0], f)

                main.glue_attention_maps(self.donorModels[0], folderToSave)
                img = QImage(f"{folderToSave}/attention_maps.png")
                pixmap = QPixmap(img).scaledToHeight(300)

                self.trainResult.setPixmap(pixmap)
                self.letsGo.setEnabled(True)

                del self.trainer_thread
                del self.trainer
            self.donorModels = []

class TrainerWorker(QObject):
    done = Signal(main.Model)
    trace_finished = Signal()

    def __init__(self, pool_size, **kwargs):
        self.pool_size = pool_size
        self.models, self.feeds = zip(*[main.construct_network(**kwargs) for _ in range(pool_size)])
        self.process_pool = futures.ThreadPoolExecutor(max_workers=os.cpu_count())
        super().__init__()

    def run_learning(self, args):
        model, feed, chosenSets = args
        datasets = []
        epoch_count = 0
        main.reset(model, feed)
        epoch_count += 1
        for path in chosenSets:
            with open(path, 'r') as f:
                datasets.append((path, [ag.aer_decode(ev) for ev in f.readline().split(' ')]))
        for n in range(model.general_parameters_set.epoch_length):
            alias, dataset = random.choice(datasets)
            feed.load(alias, dataset)
            while main.next_training_cycle(model, feed):
                pass
            self.trace_finished.emit()
        main.label_neurons(model, len(TEST_SETS) / 5)

        self.done.emit(model)


    def train(self, chosenSets):
        self.process_pool.map(self.run_learning, zip(self.models, self.feeds, [chosenSets]*self.pool_size))


class ReckognizerWorker(QObject):
    reckon = Signal(dict)
    events = Signal(list, tuple)
    done = Signal()

    def __init__(self, model, chosenSets):
        super(ReckognizerWorker, self).__init__()
        self.reckonTimer = QTimer(self)
        self.time = 0
        self.timewindow = 10
        self.fps = 50
        self.neuron_signals = {}
        self.datasets = []
        self.model = model
        for layer in self.model.layers:
            for neuron in layer["neurons"]:
                neuron.learn = False
        for path in chosenSets:
            with open(path.trace_path, 'r') as f:
                self.datasets.append((path, [ag.aer_decode(ev) for ev in f.readline().split(' ')]))
        self.feed = main.DataFeed("iter", self.model)
        main.reset(self.model, self.feed, issoft=True)
        self.feed.load(*self.datasets.pop())
        self.reckonTimer.timeout.connect(self.reckognize_frame)

        self.lut = {synapse:
                        [neuron.label for neuron in self.model.layers[-1]["neurons"]
                         if neuron.output_address == synapse][0]
                    for synapse in self.model.outputs}

    def reckognize_frame(self):
        self.events.emit(self.feed.next_events(peek=True), self.model.layers[-1].per_field_shape)
        fired_neurons = [self.lut[synapse] for synapse in self.model.outputs if self.model.state[synapse]]
        for label in fired_neurons:
            self.neuron_signals[label] = self.time
        self.neuron_signals = {k:v for k,v in self.neuron_signals.items() if v > self.time-self.timewindow}
        self.reckon.emit(self.neuron_signals)
        self.time += 1
        if not main.next_recognition_cycle(self.model, self.feed):
            if len(self.datasets):
                self.feed.load(*self.datasets.pop())
            else:
                self.done.emit()
                return
        self.reckonTimer.start(1000/self.fps)


if __name__ == "__main__":

    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
