import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFileDialog, QTextEdit, QMessageBox)
from PyQt6.QtCore import QThread, pyqtSignal
import open3d as o3d

# Import the pipeline
from reconstruct import run_pipeline

class PipelineWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str) # Emits the path to the ply file, or empty string if failed

    def __init__(self, image_dir, output_dir):
        super().__init__()
        self.image_dir = image_dir
        self.output_dir = output_dir

    def log_callback(self, message):
        self.log_signal.emit(message)

    def run(self):
        try:
            self.log_callback("--- Pipeline Started ---")
            ply_path = run_pipeline(self.image_dir, self.output_dir, log_callback=self.log_callback, show_viz=False)
            if ply_path and os.path.exists(ply_path):
                self.log_callback("--- Pipeline Completed Successfully ---")
                self.finished_signal.emit(ply_path)
            else:
                self.log_callback("--- Pipeline Failed ---")
                self.finished_signal.emit("")
        except Exception as e:
            self.log_callback(f"Exception during pipeline execution: {str(e)}")
            self.finished_signal.emit("")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Reconstruction Pipeline GUI")
        self.setMinimumSize(700, 500)
        
        self.ply_path = ""
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Input Directory Setup
        input_layout = QHBoxLayout()
        self.input_label = QLabel("Images Directory:")
        self.input_label.setFixedWidth(100)
        self.input_edit = QLineEdit(str(Path("images").absolute()))
        self.input_btn = QPushButton("Browse...")
        self.input_btn.clicked.connect(self.browse_input)
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(self.input_edit)
        input_layout.addWidget(self.input_btn)
        main_layout.addLayout(input_layout)
        
        # Output Directory Setup
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output Directory:")
        self.output_label.setFixedWidth(100)
        self.output_edit = QLineEdit(str(Path("output").absolute()))
        self.output_btn = QPushButton("Browse...")
        self.output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(self.output_btn)
        main_layout.addLayout(output_layout)
        
        # Log Box
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas;")
        main_layout.addWidget(self.log_box)
        
        # Buttons Setup
        button_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("Run Pipeline")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("background-color: #2b78e4; color: white; font-weight: bold;")
        self.run_btn.clicked.connect(self.run_pipeline)
        button_layout.addWidget(self.run_btn)
        
        self.view_btn = QPushButton("View 3D Model")
        self.view_btn.setMinimumHeight(40)
        self.view_btn.setEnabled(False)
        self.view_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.view_btn.clicked.connect(self.view_model)
        button_layout.addWidget(self.view_btn)
        
        main_layout.addLayout(button_layout)
        
        # Info Label
        info_label = QLabel("Note: True terminal output from C++ pycolmap will still print to the console window.")
        info_label.setStyleSheet("color: gray; font-style: italic;")
        main_layout.addWidget(info_label)

    def browse_input(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Images Directory", self.input_edit.text())
        if directory:
            self.input_edit.setText(directory)

    def browse_output(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_edit.text())
        if directory:
            self.output_edit.setText(directory)

    def append_log(self, text):
        self.log_box.append(text)

    def run_pipeline(self):
        img_dir = self.input_edit.text()
        out_dir = self.output_edit.text()
        
        if not os.path.exists(img_dir):
            QMessageBox.critical(self, "Error", f"Images directory does not exist:\n{img_dir}")
            return
            
        self.log_box.clear()
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running...")
        self.view_btn.setEnabled(False)
        self.ply_path = ""
        
        self.worker = PipelineWorker(img_dir, out_dir)
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(self.pipeline_finished)
        self.worker.start()

    def pipeline_finished(self, ply_path):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Pipeline")
        if ply_path:
            self.ply_path = ply_path
            self.view_btn.setEnabled(True)
            QMessageBox.information(self, "Success", "3D Pipeline completed successfully!")
        else:
            QMessageBox.warning(self, "Failed", "Pipeline failed to produce a 3D model.")

    def view_model(self):
        if not self.ply_path or not os.path.exists(self.ply_path):
            QMessageBox.critical(self, "Error", "Model file not found.")
            return
            
        try:
            pcd = o3d.io.read_point_cloud(self.ply_path)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            o3d.visualization.draw_geometries([pcd], window_name="3D Pipe Reconstruction")
        except Exception as e:
            QMessageBox.critical(self, "Visualization Error", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
