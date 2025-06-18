import multiprocessing
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import os
import sys
import shutil
import tempfile

sys.path.append(os.path.abspath("."))

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

from segmentation.total import run_TS
from converter.meshconverter_nii import nii_mask_2_stl
from converter.bodyconverter import convert_dicom_to_nifti
from radiomics.shape import run_combined_descriptor
from model.external_patient import predict_with_model

class PyramidApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pyramid(Python RadioMics-Detector)")
        yonsei = tk.PhotoImage(file = r'')
        self.root.iconphoto(False, yonsei)
        self._build_ui()

    def _build_ui(self):
        tk.Label(self.root, text="DICOM 폴더").grid(row=0, column=0, sticky="w")
        self.dicom_dir_entry = tk.Entry(self.root, width=60)
        self.dicom_dir_entry.grid(row=0, column=1)
        tk.Button(self.root, text="찾기", command=self.choose_dicom_dir).grid(row=0, column=2)

        tk.Label(self.root, text="장기 이름").grid(row=2, column=0, sticky="w")
        organ_list = ['pancreas', 'lung', 'kidney', 'liver', 'spleen']
        self.organ_combobox = ttk.Combobox(self.root, values = organ_list, width=60)
        self.organ_combobox.grid(row = 1, column = 1, columnspan =2)
        self.organ_combobox.set('장기 선택')

        self.start_button = tk.Button(self.root, text="시작", command=self.start_pipeline)
        self.start_button.grid(row=2, column=0, columnspan=3, pady=10)

        self.log_output = scrolledtext.ScrolledText(self.root, height=15, width=80, state='disabled')
        self.log_output.grid(row=3, column=0, columnspan=3, padx=5, pady=5)
        self.log('프로그램을 시작했습니다!')

    def log(self, message):
        self.log_output.config(state='normal')
        self.log_output.insert(tk.END, message + "\n")
        self.log_output.config(state='disabled')
        self.log_output.see(tk.END)

    def choose_dicom_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.dicom_dir_entry.delete(0, tk.END)
            self.dicom_dir_entry.insert(0, os.path.normpath(path))
            
    def start_pipeline(self):
        dicom_path = self.dicom_dir_entry.get()
        organ = self.organ_combobox.get().strip().lower()

        if not all([dicom_path, organ]):
            messagebox.showwarning("하나라도 빼먹으면 서운해...")
            return

        temp_dir = tempfile.mkdtemp()

        try:
            nifti_path = os.path.join(temp_dir, 'body.nii.gz')
            nifti_output = convert_dicom_to_nifti(dicom_path, nifti_path)

            ts_path = os.path.join(temp_dir, f'{organ}.nii.gz')
            ts_output = run_TS(dicom_path, ts_path, organ)

            stl_path = os.path.join(temp_dir, f"{organ}.stl")
            stl_output = nii_mask_2_stl(ts_output, stl_path)
        
            xlsx_path = os.path.join(temp_dir, f"{organ}_merged_features.xlsx")
            xlsx_output = run_combined_descriptor(nifti_output, ts_output, stl_output, xlsx_path)

            model_path = resource_path(f'best_model_fold_{organ}.pt')
            scaler_path = resource_path(f'scaler_fold_{organ}.pkl')
            predict_with_model(xlsx_output, model_path, scaler_path, log_callback=self.log)
            
        finally: 
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) > 1 and sys.argv[1] == "--prevent-loop":
        sys.exit(0)
    root = tk.Tk()
    app = PyramidApp(root)
    root.mainloop()
