import sys
import os
import cv2
import time
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QSlider, QStyle, QShortcut, 
    QSizePolicy, QComboBox 
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QKeySequence

class PoseVisualizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BlazePose Biomechanical Inspector")
        self.setGeometry(100, 100, 1024, 768)
        self.setAcceptDrops(True)

        # MediaPipe BlazePose setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Video playback state
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_next_frame)
        self.is_playing = False
        self.total_frames = 0
        self.current_frame_idx = 0
        
        # Defaulted to 60fps and added time-sync variables
        self.fps = 60.0 
        self.playback_speed = 1.0
        self.frame_accumulator = 0.0 
        self.last_clock_time = 0.0
        self.last_pixmap = None

        self.init_ui()
        self.init_shortcuts()

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Video Display Label
        self.video_label = QLabel("Drag & Drop MP4 Video Here\n— or click 'Browse Video' —", self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "border: 2px dashed #555; background-color: #121212; color: #888; font-size: 16px;"
        )
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label.setMinimumSize(400, 300)
        main_layout.addWidget(self.video_label, stretch=1)

        # Timeline Slider Row
        slider_layout = QHBoxLayout()
        self.time_label = QLabel("00:00 / 00:00", self)
        self.time_label.setStyleSheet("color: #ccc; font-family: monospace;")
        slider_layout.addWidget(self.time_label)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderMoved.connect(self.slider_moved)
        self.slider.sliderReleased.connect(self.slider_released)
        slider_layout.addWidget(self.slider)

        self.frame_label = QLabel("Frame: 0 / 0", self)
        self.frame_label.setStyleSheet("color: #ccc; font-family: monospace;")
        slider_layout.addWidget(self.frame_label)
        main_layout.addLayout(slider_layout)

        # Playback Controls Row
        controls_layout = QHBoxLayout()

        self.browse_btn = QPushButton("Browse Video", self)
        self.browse_btn.clicked.connect(self.browse_file)
        controls_layout.addWidget(self.browse_btn)

        self.step_back_btn = QPushButton("◀ Step (-1)", self)
        self.step_back_btn.clicked.connect(self.step_backward)
        self.step_back_btn.setEnabled(False)
        controls_layout.addWidget(self.step_back_btn)

        self.play_btn = QPushButton(self)
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        controls_layout.addWidget(self.play_btn)

        self.step_forward_btn = QPushButton("Step (+1) ▶", self)
        self.step_forward_btn.clicked.connect(self.step_forward)
        self.step_forward_btn.setEnabled(False)
        controls_layout.addWidget(self.step_forward_btn)

        self.restart_btn = QPushButton("Restart", self)
        self.restart_btn.clicked.connect(self.restart_video)
        self.restart_btn.setEnabled(False)
        controls_layout.addWidget(self.restart_btn)

        self.speed_combo = QComboBox(self)
        self.speed_combo.addItems(["0.25x", "0.5x", "0.75x", "1.0x", "1.5x", "2.0x", "3.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.currentTextChanged.connect(self.change_speed)
        controls_layout.addWidget(self.speed_combo)

        self.status_label = QLabel("No video loaded", self)
        self.status_label.setStyleSheet("color: #aaa;")
        controls_layout.addWidget(self.status_label)

        main_layout.addLayout(controls_layout)

    def init_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_play)
        QShortcut(QKeySequence(Qt.Key_Left), self, self.step_backward)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.step_forward)

    def change_speed(self, text):
        self.playback_speed = float(text.replace("x", ""))
        # We no longer restart the timer here. The time-sync logic handles speed automatically.

    # --- Drag & Drop ---
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if os.path.isfile(file_path):
                self.load_video(file_path)

    # --- Video Management ---
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.load_video(file_path)

    def load_video(self, file_path):
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            self.status_label.setText("Failed to open video.")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Enforce 60fps default unless the video specifically reports differently
        extracted_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = extracted_fps if extracted_fps > 0 else 60.0
        
        self.slider.setRange(0, max(0, self.total_frames - 1))

        self.play_btn.setEnabled(True)
        self.step_back_btn.setEnabled(True)
        self.step_forward_btn.setEnabled(True)
        self.restart_btn.setEnabled(True)
        self.status_label.setText(f"Loaded: {os.path.basename(file_path)}")

        self.is_playing = False
        self.frame_accumulator = 0.0
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.render_frame_at(0, seek=True)

    # --- Playback Logic ---
    def toggle_play(self):
        if not self.cap or not self.cap.isOpened():
            return
        if self.is_playing:
            self.timer.stop()
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            # Sync the start time to the real-world clock
            self.last_clock_time = time.time()
            self.timer.start(5) # Run loop as fast as the CPU allows
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.is_playing = not self.is_playing

    def restart_video(self):
        self.frame_accumulator = 0.0
        self.render_frame_at(0, seek=True)

    def step_forward(self):
        if self.is_playing:
            self.toggle_play()
        self.frame_accumulator = 0.0
        self.render_frame_at(self.current_frame_idx + 1, seek=True)

    def step_backward(self):
        if self.is_playing:
            self.toggle_play()
        self.frame_accumulator = 0.0
        self.render_frame_at(self.current_frame_idx - 1, seek=True)

    def slider_pressed(self):
        if self.is_playing:
            self.timer.stop()

    def slider_moved(self, value):
        self.frame_accumulator = 0.0
        self.render_frame_at(value, seek=True)

    def slider_released(self):
        if self.is_playing:
            self.last_clock_time = time.time()
            self.timer.start(5)

    def play_next_frame(self):
        # Calculate exactly how much real-world time passed since the last tick
        now = time.time()
        delta_sec = now - self.last_clock_time
        self.last_clock_time = now

        # Convert elapsed time into frames based on the selected speed
        frames_to_advance = delta_sec * self.fps * self.playback_speed
        self.frame_accumulator += frames_to_advance
        
        whole_frames_to_skip = int(self.frame_accumulator)
        
        # If speed is slow (e.g. 0.25x), we wait until enough time passes to render the next frame
        if whole_frames_to_skip == 0:
            return
            
        self.frame_accumulator -= whole_frames_to_skip
        target_idx = self.current_frame_idx + whole_frames_to_skip

        if target_idx >= self.total_frames:
            self.timer.stop()
            self.is_playing = False
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            return
        
        # If we skip more than 1 frame (e.g. 2.0x speed), force OpenCV to seek
        seek_needed = (whole_frames_to_skip > 1)
        self.render_frame_at(target_idx, seek=seek_needed)

    # --- Frame Decoding & Rendering ---
    def render_frame_at(self, target_idx, seek=True):
        if self.cap is None or not self.cap.isOpened():
            return

        target_idx = max(0, min(target_idx, self.total_frames - 1))
        self.current_frame_idx = target_idx
        
        if seek:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)

        ret, frame = self.cap.read()
        if not ret:
            return

        # Update UI progress labels
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame_idx)
        self.slider.blockSignals(False)

        current_secs = int(self.current_frame_idx / self.fps)
        total_secs = int(self.total_frames / self.fps)
        self.time_label.setText(f"{current_secs//60:02d}:{current_secs%60:02d} / {total_secs//60:02d}:{total_secs%60:02d}")
        self.frame_label.setText(f"Frame: {self.current_frame_idx} / {self.total_frames}")

        # MediaPipe BlazePose inference
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.pose.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                rgb_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Convert to QPixmap and render
        h, w, ch = rgb_frame.shape
        qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.last_pixmap = QPixmap.fromImage(qt_image)
        self.update_video_label()

    def update_video_label(self):
        if self.last_pixmap:
            scaled = self.last_pixmap.scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_video_label()

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        self.pose.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PoseVisualizerApp()
    window.show()
    sys.exit(app.exec_())