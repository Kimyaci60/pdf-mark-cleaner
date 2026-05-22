import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import fitz
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Tuple, List
import logging

# =====================================
# LOGGING AYARLARI
# =====================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================
# RENKLİ İŞARETLER YAPISI
# =====================================

@dataclass
class ColorRange:
    """Renk aralığını tanımlar"""
    name: str
    lower: np.ndarray
    upper: np.ndarray
    enabled: bool = True

class ColorManager:
    """Renk yönetimi"""
    
    def __init__(self):
        # HSV renk aralıkları (daha geniş ve hassas)
        self.colors = {
            'red': ColorRange(
                'Kırmızı',
                np.array([0, 30, 30]),
                np.array([20, 255, 255])
            ),
            'red2': ColorRange(
                'Kırmızı 2',
                np.array([160, 30, 30]),
                np.array([180, 255, 255])
            ),
            'pink': ColorRange(
                'Pembe',
                np.array([140, 25, 25]),
                np.array([170, 255, 255])
            ),
            'yellow': ColorRange(
                'Sarı',
                np.array([15, 40, 40]),
                np.array([40, 255, 255])
            ),
            'green': ColorRange(
                'Yeşil',
                np.array([35, 30, 30]),
                np.array([90, 255, 255])
            ),
            'blue': ColorRange(
                'Mavi',
                np.array([90, 30, 30]),
                np.array([130, 255, 255])
            ),
            'purple': ColorRange(
                'Mor',
                np.array([125, 30, 30]),
                np.array([155, 255, 255])
            ),
        }
    
    def get_enabled_colors(self) -> List[ColorRange]:
        """Etkin renkleri döndür"""
        return [color for color in self.colors.values() if color.enabled]
    
    def update_color(self, color_key: str, lower: np.ndarray, upper: np.ndarray):
        """Renk aralığını güncelle"""
        if color_key in self.colors:
            self.colors[color_key].lower = lower
            self.colors[color_key].upper = upper

# =====================================
# PDF İŞARETLERİ TEMİZLEYİCİ
# =====================================

class PDFMarkCleaner:
    """PDF işaretlerini temizle"""
    
    def __init__(self, color_manager: ColorManager):
        self.color_manager = color_manager
        self.is_cancelled = False
    
    def cancel_operation(self):
        """İşlemi iptal et"""
        self.is_cancelled = True
    
    def create_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        """Tüm işaretler için maske oluştur"""
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        
        for color in self.color_manager.get_enabled_colors():
            color_mask = cv2.inRange(
                hsv_image,
                color.lower,
                color.upper
            )
            mask = cv2.bitwise_or(mask, color_mask)
        
        return mask
    
    def clean_marks(self, pdf_path: str, output_path: str, 
                   progress_callback=None, quality: int = 2, 
                   threshold: float = 0.5) -> bool:
        """
        PDF'den işaretleri temizle
        
        Args:
            pdf_path: Giriş PDF yolu
            output_path: Çıkış PDF yolu
            progress_callback: İlerleme güncellemesi callback
            quality: Görüntü kalitesi (1-3)
            threshold: Temizleme eşiği (0.5-1.0)
        
        Returns:
            Başarı durumu
        """
        try:
            self.is_cancelled = False
            doc = fitz.open(pdf_path)
            yeni_pdf = fitz.open()
            
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                if self.is_cancelled:
                    logger.info("İşlem kullanıcı tarafından iptal edildi")
                    doc.close()
                    yeni_pdf.close()
                    return False
                
                try:
                    page = doc[page_num]
                    
                    # Görüntü al (kaliteye göre çarpan)
                    matrix = fitz.Matrix(quality, quality)
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    
                    # Numpy dizisine dönüştür
                    img = np.frombuffer(
                        pix.samples,
                        dtype=np.uint8
                    ).reshape(pix.height, pix.width, 3).copy()
                    
                    # RGB -> HSV (daha iyi renk algılaması)
                    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                    
                    # Maske oluştur
                    mask = self.create_mask(hsv)
                    
                    # Gürültü temizle (daha agresif)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    mask = cv2.morphologyEx(
                        mask,
                        cv2.MORPH_OPEN,
                        kernel,
                        iterations=2
                    )
                    mask = cv2.morphologyEx(
                        mask,
                        cv2.MORPH_CLOSE,
                        kernel,
                        iterations=1
                    )
                    
                    # Dilate - işaretleri biraz genişlet
                    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mask = cv2.dilate(mask, kernel_dilate, iterations=1)
                    
                    # Beyaza boya (daha güçlü)
                    img[mask > 100] = [255, 255, 255]
                    
                    # Temp PNG
                    temp_file = tempfile.NamedTemporaryFile(
                        suffix=".png",
                        delete=False
                    )
                    temp_path = temp_file.name
                    
                    # PNG olarak kaydet (lossless)
                    Image.fromarray(img).save(temp_path)
                    
                    # PDF sayfası
                    new_page = yeni_pdf.new_page(
                        width=pix.width,
                        height=pix.height
                    )
                    
                    rect = fitz.Rect(0, 0, pix.width, pix.height)
                    new_page.insert_image(rect, filename=temp_path)
                    
                    # Temizle
                    temp_file.close()
                    os.unlink(temp_path)
                    
                    # İlerleme güncelle
                    if progress_callback:
                        progress_callback(page_num + 1, total_pages)
                    
                    logger.info(f"Sayfa {page_num + 1}/{total_pages} işlendi")
                
                except Exception as e:
                    logger.error(f"Sayfa {page_num + 1} işlenirken hata: {str(e)}")
                    continue
            
            # PDF'yi kaydet
            yeni_pdf.save(
                output_path,
                garbage=4,
                deflate=True
            )
            
            yeni_pdf.close()
            doc.close()
            
            logger.info(f"PDF başarıyla kaydedildi: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"PDF işlenirken hata: {str(e)}")
            raise

# =====================================
# GUI UYGULAMASI
# =====================================

class PDFCleanerApp:
    """Ana uygulama sınıfı"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("PDF İşaret Temizleyici v2.1")
        self.root.geometry("700x600")
        self.root.resizable(True, True)
        
        self.color_manager = ColorManager()
        self.cleaner = PDFMarkCleaner(self.color_manager)
        self.is_processing = False
        
        self.setup_gui()
    
    def setup_gui(self):
        """GUI oluştur"""
        
        # Ana Canvas ve Scrollbar
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Başlık
        title = tk.Label(
            scrollable_frame,
            text="PDF İşaret Temizleyici",
            font=("Arial", 16, "bold"),
            fg="#2c3e50"
        )
        title.pack(pady=10)
        
        # PDF Seçim Frame
        frame1 = tk.LabelFrame(
            scrollable_frame,
            text="PDF Seç",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=8
        )
        frame1.pack(padx=12, pady=8, fill="x")
        
        self.entry = tk.Entry(
            frame1,
            width=50,
            font=("Arial", 8)
        )
        self.entry.pack(side="left", fill="x", expand=True, padx=3)
        
        btn_select = tk.Button(
            frame1,
            text="Gözat",
            width=8,
            command=self.pdf_sec,
            bg="#3498db",
            fg="white",
            font=("Arial", 8)
        )
        btn_select.pack(side="left", padx=3)
        
        # Renk Seçim Frame
        frame2 = tk.LabelFrame(
            scrollable_frame,
            text="Temizlenecek Renkler",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=8
        )
        frame2.pack(padx=12, pady=8, fill="x")
        
        self.color_vars = {}
        colors_grid = tk.Frame(frame2)
        colors_grid.pack()
        
        for i, (key, color) in enumerate(self.color_manager.colors.items()):
            row = i // 3
            col = i % 3
            
            var = tk.BooleanVar(value=color.enabled)
            self.color_vars[key] = var
            
            check = tk.Checkbutton(
                colors_grid,
                text=color.name,
                variable=var,
                command=lambda k=key: self.update_color_status(k),
                font=("Arial", 8)
            )
            check.grid(row=row, column=col, sticky="w", padx=10, pady=3)
        
        # Ayarlar Frame
        frame3 = tk.LabelFrame(
            scrollable_frame,
            text="İşlem Ayarları",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=8
        )
        frame3.pack(padx=12, pady=8, fill="x")
        
        # Kalite
        quality_frame = tk.Frame(frame3)
        quality_frame.pack(fill="x", pady=4)
        
        tk.Label(
            quality_frame,
            text="Görüntü Kalitesi:",
            font=("Arial", 8),
            width=15,
            anchor="w"
        ).pack(side="left", padx=3)
        
        self.quality_var = tk.IntVar(value=2)
        quality_scale = tk.Scale(
            quality_frame,
            from_=1,
            to=3,
            orient="horizontal",
            variable=self.quality_var,
            length=150,
            font=("Arial", 7)
        )
        quality_scale.pack(side="left", padx=3)
        
        tk.Label(
            quality_frame,
            text="1=Hızlı, 3=Yüksek",
            font=("Arial", 7),
            fg="gray"
        ).pack(side="left", padx=3)
        
        # Threshold (Duyarlılık)
        threshold_frame = tk.Frame(frame3)
        threshold_frame.pack(fill="x", pady=4)
        
        tk.Label(
            threshold_frame,
            text="Temizleme Duyarlılığı:",
            font=("Arial", 8),
            width=15,
            anchor="w"
        ).pack(side="left", padx=3)
        
        self.threshold_var = tk.DoubleVar(value=0.5)
        threshold_scale = tk.Scale(
            threshold_frame,
            from_=0.3,
            to=1.0,
            orient="horizontal",
            variable=self.threshold_var,
            resolution=0.1,
            length=150,
            font=("Arial", 7)
        )
        threshold_scale.pack(side="left", padx=3)
        
        tk.Label(
            threshold_frame,
            text="0.3=Agresif, 1.0=Hafif",
            font=("Arial", 7),
            fg="gray"
        ).pack(side="left", padx=3)
        
        # İlerleme Çubuğu
        progress_frame = tk.Frame(frame3)
        progress_frame.pack(fill="x", pady=4)
        
        self.progress = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=300
        )
        self.progress.pack(fill="x", padx=3)
        
        self.progress_label = tk.Label(
            progress_frame,
            text="Hazır",
            font=("Arial", 7),
            fg="gray"
        )
        self.progress_label.pack()
        
        # Bilgi Frame
        frame_info = tk.LabelFrame(
            scrollable_frame,
            text="Bilgilendirme",
            font=("Arial", 9, "bold"),
            padx=8,
            pady=6
        )
        frame_info.pack(padx=12, pady=8, fill="both")
        
        self.info_text = tk.Text(
            frame_info,
            height=3,
            width=60,
            font=("Arial", 7),
            bg="#ecf0f1"
        )
        self.info_text.pack(fill="both")
        self.info_text.config(state="disabled")
        
        # Butonlar Frame
        frame4 = tk.Frame(scrollable_frame)
        frame4.pack(pady=10)
        
        self.btn_clean = tk.Button(
            frame4,
            text="Temizle",
            width=15,
            bg="#27ae60",
            fg="white",
            font=("Arial", 9, "bold"),
            command=self.temizle_thread,
            cursor="hand2"
        )
        self.btn_clean.pack(side="left", padx=3)
        
        self.btn_cancel = tk.Button(
            frame4,
            text="İptal",
            width=15,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 9, "bold"),
            command=self.cancel_operation,
            cursor="hand2",
            state="disabled"
        )
        self.btn_cancel.pack(side="left", padx=3)
        
        self.add_info("✓ Uygulama hazır\n✓ PDF seçin ve Temizle'ye tıklayın")
    
    def add_info(self, text: str):
        """Bilgi ekle"""
        self.info_text.config(state="normal")
        self.info_text.insert(tk.END, text + "\n")
        self.info_text.see(tk.END)
        self.info_text.config(state="disabled")
    
    def pdf_sec(self):
        """PDF dosyası seç"""
        dosya = filedialog.askopenfilename(
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        
        if dosya:
            self.entry.delete(0, tk.END)
            self.entry.insert(0, dosya)
            self.add_info(f"✓ Seçilen: {os.path.basename(dosya)}")
    
    def update_color_status(self, color_key: str):
        """Renk durumunu güncelle"""
        is_enabled = self.color_vars[color_key].get()
        self.color_manager.colors[color_key].enabled = is_enabled
    
    def update_progress(self, current: int, total: int):
        """İlerleme çubuğunu güncelle"""
        progress_percent = (current / total) * 100
        self.progress['value'] = progress_percent
        self.progress_label.config(
            text=f"{current}/{total} sayfa ({progress_percent:.0f}%)"
        )
        self.root.update_idletasks()
    
    def cancel_operation(self):
        """İşlemi iptal et"""
        self.cleaner.cancel_operation()
        self.btn_cancel.config(state="disabled")
        self.add_info("⚠ İptal ediliyor...")
    
    def temizle_thread(self):
        """Temizleme işlemini başlat (thread'de)"""
        pdf_path = self.entry.get()
        
        if not pdf_path:
            messagebox.showerror("Hata", "Lütfen bir PDF dosyası seçin")
            return
        
        if not os.path.exists(pdf_path):
            messagebox.showerror("Hata", "Seçilen PDF dosyası bulunamadı")
            return
        
        if not self.color_vars or not any(self.color_vars.values()):
            messagebox.showwarning("Uyarı", "Lütfen en az bir renk seçin")
            return
        
        # Thread'de çalıştır
        thread = threading.Thread(target=self.temizle)
        thread.daemon = True
        thread.start()
    
    def temizle(self):
        """Asenkron temizleme işlemi"""
        try:
            self.is_processing = True
            self.btn_clean.config(state="disabled")
            self.btn_cancel.config(state="normal")
            self.progress['value'] = 0
            
            pdf_path = self.entry.get()
            output_path = os.path.splitext(pdf_path)[0] + "_temiz.pdf"
            quality = self.quality_var.get()
            threshold = self.threshold_var.get()
            
            self.add_info("⏳ İşlem başlatılıyor...")
            self.root.update()
            
            # Temizleme işlemi
            success = self.cleaner.clean_marks(
                pdf_path,
                output_path,
                progress_callback=self.update_progress,
                quality=quality,
                threshold=threshold
            )
            
            if success:
                self.progress['value'] = 100
                self.progress_label.config(text="✓ Tamamlandı!")
                
                self.add_info(f"✓ Başarılı!\n✓ {os.path.basename(output_path)}")
                
                messagebox.showinfo(
                    "Başarılı ✓",
                    f"PDF başarıyla temizlendi!\n\n{output_path}"
                )
                
                logger.info(f"İşlem tamamlandı: {output_path}")
            else:
                self.add_info("⚠ İşlem iptal edildi")
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Temizleme hatası: {error_msg}")
            self.add_info(f"✗ Hata: {error_msg[:40]}")
            messagebox.showerror("Hata ✗", f"Hata: {error_msg}")
        
        finally:
            self.is_processing = False
            self.btn_clean.config(state="normal")
            self.btn_cancel.config(state="disabled")

# =====================================
# MAIN
# =====================================

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFCleanerApp(root)
    root.mainloop()
