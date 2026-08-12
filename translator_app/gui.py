# -*- coding: utf-8 -*-
"""
批量翻译工具 - Windows 图形界面
中文 -> 俄语 / 西语 / 葡语，支持图片与视频批量翻译。
"""
import os
import sys
import threading
import queue
import webbrowser

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QProgressBar,
    QPlainTextEdit, QFileDialog, QGroupBox, QMessageBox,
)
from PySide6.QtCore import Qt, QTimer

from core.batch import run_batch
from core.translate_engine import SUPPORTED_TARGETS, _read_config_key, save_config_key

APP_TITLE = "批量翻译工具 v1.5.1（中文 → 14 种主流语言）"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(760, 640)
        self._worker = None
        self._cancel = False
        self._log_queue = queue.Queue()
        self._progress_queue = queue.Queue()
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_queues)
        self._timer.start(120)

    # ---------- UI ----------

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # 输入/输出目录
        dir_box = QGroupBox("文件夹")
        dir_layout = QVBoxLayout(dir_box)
        self.input_edit, row1 = self._dir_row("导入文件夹（含图片/视频）：", self._pick_input)
        dir_layout.addLayout(row1)
        self.output_edit, row2 = self._dir_row("导出文件夹（默认按语言生成子文件夹）：", self._pick_output)
        dir_layout.addLayout(row2)
        layout.addWidget(dir_box)

        # 选项
        opt_box = QGroupBox("选项")
        opt_layout = QVBoxLayout(opt_box)
        self.lang_cbs = {}
        default_on = {"ru", "es", "pt", "en"}
        lang_items = list(SUPPORTED_TARGETS.items())
        for row_start in range(0, len(lang_items), 5):
            lang_row = QHBoxLayout()
            if row_start == 0:
                lang_row.addWidget(QLabel("目标语言："))
            else:
                lang_row.addWidget(QLabel("　"))
            for code, name in lang_items[row_start:row_start + 5]:
                cb = QCheckBox(name)
                cb.setChecked(code in default_on)
                self.lang_cbs[code] = cb
                lang_row.addWidget(cb)
            lang_row.addStretch(1)
            opt_layout.addLayout(lang_row)

        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("智谱AI Key（可选，填了翻译质量更高）："))
        self.key_edit = QLineEdit()
        self.key_edit.setPlaceholderText("在 bigmodel.cn 免费申请，留空则用免费机翻")
        self.key_edit.setText(_read_config_key())
        key_row.addWidget(self.key_edit, 1)
        opt_layout.addLayout(key_row)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("处理内容："))
        self.cb_img = QCheckBox("图片（抹字+回填排版）")
        self.cb_img.setChecked(True)
        self.cb_vid = QCheckBox("视频（字幕+AI配音）")
        self.cb_vid.setChecked(True)
        self.cb_bg = QCheckBox("视频保留背景音（人声分离，较慢）")
        self.cb_bg.setChecked(True)
        self.cb_cover = QCheckBox("去除视频画面中的中文并替换为译文（逐帧处理，较慢）")
        self.cb_cover.setChecked(True)
        type_row.addWidget(self.cb_img)
        type_row.addWidget(self.cb_vid)
        type_row.addWidget(self.cb_bg)
        type_row.addWidget(self.cb_cover)
        type_row.addStretch(1)
        opt_layout.addLayout(type_row)
        layout.addWidget(opt_box)

        # 进度
        prog_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.btn_start = QPushButton("开始翻译")
        self.btn_start.clicked.connect(self._start)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_task)
        self.btn_open = QPushButton("打开导出文件夹")
        self.btn_open.clicked.connect(self._open_output)
        prog_row.addWidget(self.progress, 1)
        prog_row.addWidget(self.btn_start)
        prog_row.addWidget(self.btn_cancel)
        prog_row.addWidget(self.btn_open)
        layout.addLayout(prog_row)

        # 日志
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, 1)

        self._log("提示：选择包含图片/视频的文件夹，点击“开始翻译”。\n"
                  "首次运行会自动下载语音识别模型（约 500MB，只需一次）。\n"
                  "导出结果按语言分文件夹：俄语 / 西语 / 葡语。")

    def _dir_row(self, label, on_pick):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        edit = QLineEdit()
        row.addWidget(edit, 1)
        btn = QPushButton("浏览…")
        btn.clicked.connect(on_pick)
        row.addWidget(btn)
        return edit, row

    # ---------- 事件 ----------

    def _pick_input(self):
        d = QFileDialog.getExistingDirectory(self, "选择导入文件夹")
        if d:
            self.input_edit.setText(d)
            if not self.output_edit.text():
                self.output_edit.setText(os.path.join(d, "翻译输出"))

    def _pick_output(self):
        d = QFileDialog.getExistingDirectory(self, "选择导出文件夹")
        if d:
            self.output_edit.setText(d)

    def _open_output(self):
        path = self.output_edit.text().strip()
        if path and os.path.isdir(path):
            webbrowser.open(f"file:///{path.replace(os.sep, '/')}")
        else:
            QMessageBox.information(self, "提示", "导出文件夹不存在")

    @staticmethod
    def _clean_path(text):
        """去掉拖拽/粘贴带来的 file:/// 前缀"""
        t = text.strip().strip('"')
        if t.lower().startswith("file:///"):
            t = t[8:]
        return t.replace(chr(92), "/").rstrip("/")

    def _start(self):
        input_dir = self._clean_path(self.input_edit.text())
        output_dir = self._clean_path(self.output_edit.text())
        self.input_edit.setText(input_dir)
        if output_dir:
            self.output_edit.setText(output_dir)
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self, "提示", "请选择有效的导入文件夹")
            return
        if not output_dir:
            output_dir = os.path.join(input_dir, "翻译输出")
            self.output_edit.setText(output_dir)
        targets = [code for code, cb in self.lang_cbs.items() if cb.isChecked()]
        save_config_key(self.key_edit.text())
        if not targets:
            QMessageBox.warning(self, "提示", "请至少勾选一种目标语言")
            return
        if not (self.cb_img.isChecked() or self.cb_vid.isChecked()):
            QMessageBox.warning(self, "提示", "请至少勾选一种处理内容")
            return

        self._cancel = False
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)

        def log(msg):
            self._log_queue.put(str(msg))

        def progress(done, total, name):
            self._progress_queue.put((done, total, name))

        def cancel_check():
            return self._cancel

        def work():
            try:
                run_batch(input_dir, output_dir, targets,
                          do_images=self.cb_img.isChecked(),
                          do_videos=self.cb_vid.isChecked(),
                          keep_background=self.cb_bg.isChecked(),
                          cover_original=self.cb_cover.isChecked(),
                          log=log, progress=progress, cancel_check=cancel_check)
            except Exception as e:
                log(f"[错误] {e}")
            finally:
                self._log_queue.put("__FINISHED__")

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def _cancel_task(self):
        self._cancel = True
        self._log("正在取消…（当前文件处理完成后停止）")

    # ---------- 队列轮询 ----------

    def _poll_queues(self):
        while not self._log_queue.empty():
            msg = self._log_queue.get()
            if msg == "__FINISHED__":
                self.btn_start.setEnabled(True)
                self.btn_cancel.setEnabled(False)
                self._log("===== 全部任务结束 =====")
            else:
                self._log(msg)
        while not self._progress_queue.empty():
            done, total, name = self._progress_queue.get()
            self.progress.setMaximum(max(1, total))
            self.progress.setValue(done)
            self.progress.setFormat(f"%v/%m  {name}")

    def _log(self, msg):
        self.log_view.appendPlainText(msg)
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())


def main():
    # 未捕获异常弹窗显示，杜绝"点了没反应"
    import traceback
    def _excepthook(cls, exc, tb):
        try:
            QMessageBox.critical(None, "程序错误", "".join(traceback.format_exception(cls, exc, tb)))
        except Exception:
            pass
    sys.excepthook = _excepthook
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
