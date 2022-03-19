# This Python file uses the following encoding: utf-8
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QFile, QTimer, QDir, Qt, QObject, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, qRgb
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QComboBox, QSpinBox, QFileDialog, QCheckBox, \
    QVBoxLayout, QScrollArea

import AERGen as ag
import configmanager as cm
import main


class MainWindow(QWidget):
    startTraining = Signal(list)
    endTraining = Signal()

    def __init__(self):
        super(MainWindow, self).__init__()
        self.load_ui()

        self.newDatasetInputs = {
            "initialPosition": {"x": self.findChild(QSpinBox, "x0"), "y": self.findChild(QSpinBox, "y0")},
            "endPosition": {"x": self.findChild(QSpinBox, "x1"), "y": self.findChild(QSpinBox, "y1")},
            "radius": self.findChild(QSpinBox, "radius"),
            "speed": self.findChild(QSpinBox, "speed")
        }

        self.chosenSets = []

        self.db = cm.DBManager()

        self.statusbar = self.findChild(QLabel, "statusbar")

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
        self.datasetDelete = self.findChild(QPushButton, "datasetDelete")
        self.datasetDelete.clicked.connect(self._datasetDelete)

        self.chooseConf = self.findChild(QPushButton, "chooseConf")
        self.chooseConf.clicked.connect(self._chooseConf)
        self.numOfSets = self.findChild(QSpinBox, "numOfSets")
        self.poolSize = self.findChild(QSpinBox, "poolSize")
        self._poolSize = 1
        self.donorModels = []
        self.letsGo = self.findChild(QPushButton, "letsGo")
        self.letsGo.clicked.connect(self._letsGo)
        #self.letsGo.clicked.connect(self.testt)
        self.trainResult = self.findChild(QLabel, "trainResult")

        self.preview = self.findChild(QLabel, "preview")
        self.runLearning = self.findChild(QLabel, "trainResult")
        self.setsContainer = self.findChild(QVBoxLayout, "setsContainer")
        self.setsLayout = QVBoxLayout()

        self.loadTraces()

        widget = QWidget()
        widget.setLayout(self.setsLayout)

        scroll = QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)

        self.setsContainer.insertWidget(0, scroll)

        self.dataset = None
        self.network_conf = ""

        self.renderComplete = False
        self.renderTimer = None

        self.colortable = [qRgb(i, i, i) for i in range(256)]

        self.path = QDir(os.fspath(Path(__file__).resolve().parent))

    def load_ui(self):
        loader = QUiLoader()
        path = os.fspath(Path(__file__).resolve().parent / "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        loader.load(ui_file, self)
        ui_file.close()

    def setStatus(self, text):
        self.statusbar.setText(text)
        self.repaint()

    def loadTraces(self):
        self.setStatus('Начинаем загрузку')
        self.datasetCombo.clear()
        traces = self.db.read_trace_entries()
        for trace in traces:
            self.datasetCombo.addItem(trace.trace_alias, userData=trace.trace_path)
            button = QCheckBox(trace.trace_alias)
            button.stateChanged.connect(self.toggleDataSet(trace.trace_path))
            self.setsLayout.insertWidget(0, button)
        self.setStatus('Готово')

    def _datasetDelete(self):
        self.db.delete_trace_entry(self.datasetCombo.currentData())
        self.loadTraces()

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

    def generateDataset(self, time=0):
        print(1)
        self.setStatus('Начинаем генерацию траектории')
        self.dataset = []
        raw_dataset = ag.AERGen(radius=self.newDatasetInputs["radius"].value(),
                                speed=self.newDatasetInputs["speed"].value(),
                                pos_start=ag.Position(self.newDatasetInputs["initialPosition"]["x"].value(),
                                                      self.newDatasetInputs["initialPosition"]["y"].value()),
                                pos_end=ag.Position(self.newDatasetInputs["endPosition"]["x"].value(),
                                                    self.newDatasetInputs["endPosition"]["y"].value()),
                                start_time=time)
        for events in raw_dataset:
            for ev in events:
                self.dataset.append(ev)

    def renderPreview(self):
        self.setStatus('Начинаем показ траектории')
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
                self.setStatus('Готово')
            self.renderTimer.start(10)

    def _genNextFrame(self):
        white_square = np.ones((28, 28), dtype=np.uint8)
        current_frame = np.copy(white_square)
        frames_shown = 0
        for ev in self.dataset:
            x_coord = ev.position.x
            y_coord = ev.position.y
            color = ev.polarity * 255
            try:
                current_frame[y_coord][x_coord] = color
            except:
                pass
            frames_shown += 1
            img = QImage(current_frame, 28, 28, 28, QImage.Format_Indexed8)
            img.setColorTable(self.colortable)
            pixmap = QPixmap(img).scaledToHeight(300)
            yield pixmap

    def _appendToExisting(self):
        filename = self.path.relativeFilePath(QFileDialog.getOpenFileName(self, "Open file", "")[0])
        self._datasetLoad(path=filename)
        self.generateDataset(time=self.dataset[-1].time)
        with open(filename[0], 'a') as f:
            f.write(' '.join([ag.aer_encode(ev) for ev in self.dataset]))
        trace_entry = self.db.get_trace_by_path(filename)
        new_entry = cm.Trace(trace_entry.id, trace_entry.trace_path, trace_entry.trace_alias, trace_entry.speed,
                             self.dataset[-1].time, "")
        self.db.update(new_entry)
        self.setStatus(f'Траектория добавлена к файлу {filename}')

    def _saveAsNew(self):
        filename = QFileDialog.getSaveFileName(self, "Save file", "", ".bin")
        self.generateDataset()
        with open(''.join(filename), 'w') as f:
            f.write(' '.join([ag.aer_encode(ev)[0] for ev in self.dataset]))
            self.db.add_trace_entry(filename[0].split('/')[-1],
                                    trace_path=self.path.relativeFilePath(''.join(filename)),
                                    target_speed=self.newDatasetInputs["speed"].value(), end_time=self.dataset[-1].time)
        self.loadTraces()
        self.setStatus(f'Сохранено в файл {filename}')

    def _datasetLoad(self, path=""):
        if not path:
            path = self.datasetCombo.currentData()
        with open(path, 'r') as f:
            self.dataset = [ag.aer_decode(ev) for ev in f.readline().split(' ')]
        self.datasetPreview.setEnabled(True)
        self.setStatus(f'Загружен файл {path}')

    def _chooseConf(self):
        self.network_conf = self.path.relativeFilePath(
            QFileDialog.getOpenFileName(self, "Выберите файл с конфигурацией модели", "")[0])

    # noinspection PyUnresolvedReferences
    def _letsGo(self):
        if not self.network_conf:
            self.setStatus("Конфигурация не выбрана!")
            return
        self.letsGo.setEnabled(False)
        self._poolSize = self.poolSize.value()
        neuron_changes = {}
        general_changes = {"pool_size": str(self._poolSize)}
        self.target_model, feed = main.construct_network(feed_type="iter",
                                                         source_file=self.network_conf,
                                                         learn=False,
                                                         update_neuron_parameters=neuron_changes,
                                                         update_general_parameters=general_changes)

        self.threads = []
        self.trainers = []
        for _ in range(self._poolSize):
            thread = QThread()
            thread.start()
            trainer = TrainerWorker(feed_type="iter",
                                    source_file=self.network_conf,
                                    learn=True,
                                    update_neuron_parameters=neuron_changes,
                                    update_general_parameters=general_changes)
            trainer.moveToThread(thread)
            self.startTraining.connect(trainer.train)
            trainer.done.connect(self._trainingFinished)
            self.endTraining.connect(thread.quit)

            self.threads.append(thread)
            self.trainers.append(trainer)

        self.startTraining.emit(self.chosenSets)
        self.setStatus('Начинаем обучение')

    # noinspection PyUnresolvedReferences
    def _trainingFinished(self, donor_model):
        print("training finished")
        self.donorModels.append(donor_model)
        self._poolSize -= 1
        self.setStatus(f"Обучение сети {self.poolSize.value() - self._poolSize} из {self.poolSize.value()} закончено")
        if not self._poolSize:
            labels = main.fill_model_from_pool(self.target_model, self.donorModels)
            if len(labels) == len(self.chosenSets):
                self.endTraining.emit()
                folderToSave = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения файлов", "")
                main.save_attention_maps(self.target_model, f"experiments_results/{folderToSave}")

                with open(f"experiments_results/{folderToSave}/model.pkl", "wb") as f:
                    pickle.dump(self.target_model, f)

                glued_maps = main.glue_attention_maps(self.target_model)
                img = QImage(glued_maps, glued_maps.shape[1], glued_maps.shape[0], glued_maps.shape[1],
                             QImage.Format_Indexed8)
                img.setColorTable(self.colortable)
                pixmap = QPixmap(img).scaledToHeight(300)

                self.trainResult.setPixmap(pixmap)
                self.letsGo.setEnabled(True)
            else:
                self._poolSize = self.poolSize.value()
                self.startTraining.emit()
                self.setStatus('Начинаем обучение заново')

    def testt(self):
        if not self.network_conf:
            self.setStatus("Конфигурация не выбрана!")
            return
        self.letsGo.setEnabled(False)
        self._poolSize = self.poolSize.value()
        neuron_changes = {}
        general_changes = {"pool_size": str(self._poolSize)}
        self.target_model, feed = main.construct_network(feed_type="iter",
                                                         source_file=self.network_conf,
                                                         learn=False,
                                                         update_neuron_parameters=neuron_changes,
                                                         update_general_parameters=general_changes)

        print("training started")
        main.reset(self.target_model, feed)
        datasets = []
        epoch_count = 0

        main.reset(self.target_model, feed)
        print("model reset")
        epoch_count += 1
        print(f"epoch {epoch_count}")
        for path in self.chosenSets:
            with open(path, 'r') as f:
                datasets.append((path, [ag.aer_decode(ev) for ev in f.readline().split(' ')]))

        for n in range(self.target_model.general_parameters_set.epoch_length):
            alias, dataset = random.choice(datasets)
            feed.load(alias, dataset)
            print(f"current dataset {alias}")
            while main.next_training_cycle(self.target_model, feed):
                print(self.target_model.time)
                pass
            print("dataset finished")

        print(main.label_neurons(self.target_model))


        self.setStatus('Начинаем обучение')



class TrainerWorker(QObject):
    done = Signal(main.Model)

    def __init__(self, **kwargs):
        self.model, self.feed = main.construct_network(**kwargs)
        super().__init__()

    # noinspection PyUnresolvedReferences
    def train(self, chosenSets):
        print("training started")
        main.reset(self.model, self.feed)
        datasets = []
        epoch_count = 0

        main.reset(self.model, self.feed)
        print("model reset")
        epoch_count += 1
        print(f"epoch {epoch_count}")
        for path in chosenSets:
            with open(path, 'r') as f:
                datasets.append((path, [ag.aer_decode(ev) for ev in f.readline().split(' ')]))

        for n in range(self.model.general_parameters_set.epoch_length):
            alias, dataset = random.choice(datasets)
            self.feed.load(alias, dataset)
            print(f"current dataset {alias}")
            while True:
                print(model.time)
                t = main.next_training_cycle(self.model, self.feed)
                print(model.time)
                pass
            print("dataset finished")

        print(main.label_neurons(self.model))

        self.done.emit(self.model)


class ReckognizerWorker(QObject):
    reckon = Signal(str)

    def __init__(self, source):
        super(ReckognizerWorker, self).__init__()
        self.model = main.load_network(source)

    # noinspection PyUnresolvedReferences
    def reckognize(self, events):
        self.model.time = events[0].time
        for ev in events:
            self.model.state[ev.address] = 1

        for layer in self.model.layers:
            main.layer_update(self.model, layer)

        for synapse in self.model.outputs:
            if self.model.state[synapse]:
                label = [n.label for n in self.model.layers[-1]["neurons"] if n.output == synapse][0]
                self.reckon.emit(label)

    def reckonloop(self, feed_type, source):
        feed = main.DataFeed(feed_type, self.model)
        feed.load(source)
        while not feed.terminate:
            self.reckognize(feed.next_events())


if __name__ == "__main__":
    sys._excepthook = sys.excepthook


    def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)


    sys.excepthook = exception_hook

    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
