import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import os
import threading
from ultralytics import YOLO
from model_depth import TSTDepthModel3, TSTConfig
from torchvision import transforms as T
from PIL import Image as PILImage
from scipy import ndimage


class DepthOutlierRemover:
    """Depth outlier removal management class"""

    @staticmethod
    def iqr_filter(depth_region, multiplier=1.5):
        """IQR method - Robust against outliers"""
        depth_flat = depth_region.flatten()
        if len(depth_flat) < 5:
            return depth_flat, np.ones_like(depth_region, dtype=bool)

        Q1 = np.percentile(depth_flat, 25)
        Q3 = np.percentile(depth_flat, 75)
        IQR = Q3 - Q1

        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR

        mask = (depth_region >= lower) & (depth_region <= upper)
        return depth_region[mask], mask

    @staticmethod
    def zscore_filter(depth_region, threshold=2.0):
        """Z-Score method - For normally distributed data"""
        depth_flat = depth_region.flatten()
        if len(depth_flat) < 10:
            return depth_flat, np.ones_like(depth_region, dtype=bool)

        mean = np.mean(depth_flat)
        std = np.std(depth_flat)

        if std < 1e-6:
            return depth_flat, np.ones_like(depth_region, dtype=bool)

        z_scores = np.abs((depth_region - mean) / std)
        mask = z_scores <= threshold
        return depth_region[mask], mask

    @staticmethod
    def trimmed_filter(depth_region, lower_pct=5, upper_pct=95):
        """Percentage-based removal method - Very fast"""
        depth_flat = depth_region.flatten()
        lower = np.percentile(depth_flat, lower_pct)
        upper = np.percentile(depth_flat, upper_pct)

        mask = (depth_region >= lower) & (depth_region <= upper)
        return depth_region[mask], mask

    @staticmethod
    def adaptive_filter(depth_region):
        """Adaptive method - Selects the best method based on data"""
        depth_flat = depth_region.flatten()

        if len(depth_flat) < 10:
            return depth_flat, np.ones_like(depth_region, dtype=bool)

        # Calculate coefficient of variation to determine data dispersion
        cv = np.std(depth_flat) / (np.mean(depth_flat) + 1e-6)

        if cv > 0.3:  # High dispersion
            return DepthOutlierRemover.iqr_filter(depth_region, multiplier=1.8)
        elif cv > 0.15:  # Medium dispersion
            return DepthOutlierRemover.zscore_filter(depth_region, threshold=2.0)
        else:  # Low dispersion
            return DepthOutlierRemover.trimmed_filter(depth_region, lower_pct=2, upper_pct=98)

    @staticmethod
    def get_robust_depth_stats(depth_region, method='adaptive'):
        """
        Calculate depth statistics after outlier removal

        Returns:
            dict: contains mean, median, std, min, max, valid_ratio, outlier_removed
        """
        if method == 'iqr':
            filtered, mask = DepthOutlierRemover.iqr_filter(depth_region)
        elif method == 'zscore':
            filtered, mask = DepthOutlierRemover.zscore_filter(depth_region)
        elif method == 'trimmed':
            filtered, mask = DepthOutlierRemover.trimmed_filter(depth_region)
        else:  # adaptive
            filtered, mask = DepthOutlierRemover.adaptive_filter(depth_region)

        if len(filtered) == 0:
            return {
                'mean': 0, 'median': 0, 'std': 0,
                'min': 0, 'max': 0, 'valid_ratio': 0,
                'outlier_removed': 0, 'mask': np.ones_like(depth_region, dtype=bool)
            }

        return {
            'mean': float(np.mean(filtered)),
            'median': float(np.median(filtered)),
            'std': float(np.std(filtered)),
            'min': float(np.min(filtered)),
            'max': float(np.max(filtered)),
            'valid_ratio': len(filtered) / depth_region.size,
            'outlier_removed': depth_region.size - len(filtered),
            'mask': mask
        }


class DepthYOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv13 + Depth Estimation - Object Detection with Distance")
        self.root.geometry("1400x800")
        self.root.configure(bg='#2b2b2b')

        # Variables
        self.current_image_path = None
        self.original_image = None
        self.result_image = None
        self.depth_map = None
        self.detections = None

        # Load models in background
        self.models_loaded = False
        self.yolo_model = None
        self.depth_model = None

        # Outlier settings
        self.outlier_method = tk.StringVar(value="adaptive")
        self.show_outlier_info = tk.BooleanVar(value=True)

        # Build UI
        self.setup_ui()

        # Start loading models
        self.load_models_async()

    def setup_ui(self):
        """Create all UI elements"""

        # Main frame
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ========== Left Panel (Controls) ==========
        left_panel = tk.Frame(main_frame, bg='#3c3c3c', width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # Title
        title_label = tk.Label(left_panel, text="🎯 YOLOv13 + Depth",
                               font=('Arial', 16, 'bold'),
                               bg='#3c3c3c', fg='white')
        title_label.pack(pady=20)

        # Model status
        self.status_frame = tk.Frame(left_panel, bg='#3c3c3c')
        self.status_frame.pack(pady=10, padx=20, fill=tk.X)

        self.yolo_status = tk.Label(self.status_frame, text="⏳ Loading YOLO...",
                                    bg='#3c3c3c', fg='yellow', font=('Arial', 10))
        self.yolo_status.pack(anchor=tk.W, pady=2)

        self.depth_status = tk.Label(self.status_frame, text="⏳ Loading Depth...",
                                     bg='#3c3c3c', fg='yellow', font=('Arial', 10))
        self.depth_status.pack(anchor=tk.W, pady=2)

        # Select image button
        self.select_btn = tk.Button(left_panel, text="📁 Select Image",
                                    command=self.select_image,
                                    font=('Arial', 12), bg='#4CAF50', fg='black',
                                    height=2, cursor='hand2', state=tk.DISABLED)
        self.select_btn.pack(pady=20, padx=20, fill=tk.X)

        # Run button
        self.detect_btn = tk.Button(left_panel, text="🔍 Detect Objects & Depth",
                                    command=self.detect_objects,
                                    font=('Arial', 12), bg='#2196F3', fg='black',
                                    height=2, cursor='hand2', state=tk.DISABLED)
        self.detect_btn.pack(pady=10, padx=20, fill=tk.X)

        # ========== Outlier Removal Settings ==========
        outlier_frame = tk.LabelFrame(left_panel, text="🔧 Outlier Removal Settings",
                                      bg='#3c3c3c', fg='white', font=('Arial', 10, 'bold'))
        outlier_frame.pack(pady=20, padx=20, fill=tk.X)

        # Method selection
        method_frame = tk.Frame(outlier_frame, bg='#3c3c3c')
        method_frame.pack(pady=10, padx=10, fill=tk.X)

        tk.Label(method_frame, text="Method:", bg='#3c3c3c', fg='white').pack(anchor=tk.W)

        methods = [
            ("Adaptive (Recommended)", "adaptive"),
            ("IQR (Statistical)", "iqr"),
            ("Z-Score", "zscore"),
            ("Trimmed", "trimmed"),
            ("None (No Filter)", "none")
        ]

        for text, value in methods:
            tk.Radiobutton(method_frame, text=text, variable=self.outlier_method,
                           value=value, bg='#3c3c3c', fg='white',
                           selectcolor='#3c3c3c').pack(anchor=tk.W, pady=2)

        # Show outlier information
        self.show_info_var = tk.BooleanVar(value=True)
        tk.Checkbutton(outlier_frame, text="Show outlier removal info in results",
                       variable=self.show_info_var, bg='#3c3c3c', fg='white',
                       selectcolor='#3c3c3c').pack(anchor=tk.W, padx=10, pady=5)

        # ========== Detection Settings ==========
        settings_frame = tk.LabelFrame(left_panel, text="Detection Settings",
                                       bg='#3c3c3c', fg='white', font=('Arial', 10, 'bold'))
        settings_frame.pack(pady=10, padx=20, fill=tk.X)

        # Confidence
        conf_frame = tk.Frame(settings_frame, bg='#3c3c3c')
        conf_frame.pack(pady=10, padx=10, fill=tk.X)

        tk.Label(conf_frame, text="Confidence Threshold:",
                 bg='#3c3c3c', fg='white').pack(anchor=tk.W)

        self.confidence_var = tk.DoubleVar(value=0.5)
        self.confidence_slider = tk.Scale(conf_frame, from_=0.1, to=0.9,
                                          orient=tk.HORIZONTAL, variable=self.confidence_var,
                                          bg='#3c3c3c', fg='white', resolution=0.05)
        self.confidence_slider.pack(fill=tk.X)

        # Results display
        results_frame = tk.LabelFrame(left_panel, text="Results",
                                      bg='#3c3c3c', fg='white', font=('Arial', 14, 'bold'))
        results_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # Frame for scroll
        text_frame = tk.Frame(results_frame, bg='#3c3c3c')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.results_text = tk.Text(text_frame, height=15, width=35,
                                    bg='#2b2b2b', fg='white', font=('Arial', 14))
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar for results
        scrollbar = tk.Scrollbar(text_frame, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

        # Save button
        self.save_btn = tk.Button(left_panel, text="💾 Save Result",
                                  command=self.save_result,
                                  font=('Arial', 11), bg='#2b2b2b', fg='black',
                                  height=1, cursor='hand2', state=tk.DISABLED)
        self.save_btn.pack(pady=10, padx=20, fill=tk.X)

        # ========== Right Panel (Image Display) ==========
        right_panel = tk.Frame(main_frame, bg='#2b2b2b')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Original image
        original_frame = tk.LabelFrame(right_panel, text="Original Image",
                                       bg='#2b2b2b', fg='white', font=('Arial', 12, 'bold'))
        original_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.original_label = tk.Label(original_frame, bg='#1e1e1e', text="No image selected")
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Result image
        result_frame = tk.LabelFrame(right_panel, text="Detection Result",
                                     bg='#2b2b2b', fg='white', font=('Arial', 12, 'bold'))
        result_frame.pack(fill=tk.BOTH, expand=True)

        self.result_label = tk.Label(result_frame, bg='#1e1e1e', text="Run detection to see results")
        self.result_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def load_models_async(self):
        """Load models in a separate thread"""

        def load():
            try:
                # Load YOLO
                self.root.after(0, lambda: self.yolo_status.config(text="⏳ Loading YOLOv13...", fg='yellow'))
                self.yolo_model = YOLO('yolov13s.pt')
                self.root.after(0, lambda: self.yolo_status.config(text="✅ YOLOv13 Loaded", fg='green'))

                # Load depth model
                self.root.after(0, lambda: self.depth_status.config(text="⏳ Loading Depth Model...", fg='yellow'))
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                config = TSTConfig(max_depth=10.0)
                self.depth_model = TSTDepthModel3(config).to(device)
                checkpoint = torch.load("best_tst_depth.pth", map_location=device)
                self.depth_model.load_state_dict(checkpoint["model"])
                self.depth_model.eval()
                self.root.after(0, lambda: self.depth_status.config(text="✅ Depth Model Loaded", fg='green'))

                self.models_loaded = True
                self.root.after(0, lambda: self.select_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.detect_btn.config(state=tk.NORMAL))

            except Exception as e:
                self.root.after(0, lambda: self.yolo_status.config(text=f"❌ Error: {str(e)}", fg='red'))

        thread = threading.Thread(target=load)
        thread.daemon = True
        thread.start()

    def select_image(self):
        """Select image by user"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )

        if file_path:
            self.current_image_path = file_path
            self.original_image = cv2.imread(file_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

            # Display image in GUI
            self.display_image(self.original_image, self.original_label)

            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            self.result_label.config(image='', text="Run detection to see results")
            self.save_btn.config(state=tk.DISABLED)

    def display_image(self, image, label, max_size=(600, 400)):
        """Display image in label with size constraints"""
        if image is None:
            return

        # Resize image for display
        h, w = image.shape[:2]
        ratio = min(max_size[0] / w, max_size[1] / h)
        new_size = (int(w * ratio), int(h * ratio))
        resized = cv2.resize(image, new_size)

        # Convert to PhotoImage
        image_pil = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(image_pil)

        label.config(image=photo)
        label.image = photo  # Keep reference

    def detect_objects(self):
        """Run object detection and depth estimation with outlier removal"""
        if not self.models_loaded:
            messagebox.showwarning("Wait", "Models are still loading...")
            return

        if self.current_image_path is None:
            messagebox.showwarning("No Image", "Please select an image first")
            return

        # Disable buttons during processing
        self.detect_btn.config(state=tk.DISABLED, text="⏳ Processing...")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Processing... Please wait\n")
        self.root.update()

        # Run in separate thread
        def process():
            try:
                # Read image
                img_bgr = cv2.imread(self.current_image_path)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w = img_bgr.shape[:2]

                # Step 1: YOLO detection
                conf_threshold = self.confidence_var.get()
                yolo_results = self.yolo_model(self.current_image_path, conf=conf_threshold)

                self.detections = []
                if yolo_results[0].boxes is not None:
                    for box in yolo_results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        self.detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'class': self.yolo_model.names[int(box.cls)],
                            'confidence': float(box.conf)
                        })

                # Step 2: Depth calculation with outlier removal
                outlier_method = self.outlier_method.get()

                if self.detections and outlier_method != 'none':
                    # Prepare for depth model
                    preprocess = T.Compose([
                        T.Resize((480, 640)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

                    pil_img = PILImage.fromarray(img_rgb)
                    depth_input = preprocess(pil_img).unsqueeze(0)

                    device = next(self.depth_model.parameters()).device
                    depth_input = depth_input.to(device)

                    with torch.no_grad():
                        depth_output = self.depth_model(depth_input)
                        depth_map = depth_output["pred_d"][0, 0].cpu().numpy()

                    # Upscale depth map
                    depth_map_full = cv2.resize(depth_map, (w, h))

                    # Calculate depth for each object with outlier removal
                    outlier_remover = DepthOutlierRemover()

                    for det in self.detections:
                        x1, y1, x2, y2 = det['bbox']
                        depth_region = depth_map_full[y1:y2, x1:x2]

                        if outlier_method == 'none':
                            # Without outlier removal
                            det['depth_mean'] = float(np.mean(depth_region)) if depth_region.size > 0 else 0
                            det['depth_median'] = float(np.median(depth_region)) if depth_region.size > 0 else 0
                            det['outlier_removed'] = 0
                            det['valid_ratio'] = 1.0
                        else:
                            # With outlier removal
                            depth_stats = outlier_remover.get_robust_depth_stats(depth_region, method=outlier_method)
                            det['depth_mean'] = depth_stats['mean']
                            det['depth_median'] = depth_stats['median']
                            det['depth_min'] = depth_stats['min']
                            det['depth_max'] = depth_stats['max']
                            det['outlier_removed'] = depth_stats['outlier_removed']
                            det['valid_ratio'] = depth_stats['valid_ratio']
                            det['depth_std'] = depth_stats['std']

                elif self.detections:
                    # Fallback: if depth model not available
                    for det in self.detections:
                        det['depth_mean'] = 0
                        det['depth_median'] = 0
                        det['outlier_removed'] = 0
                        det['valid_ratio'] = 0

                # ========== Step 3: Create output image ==========
                # Only display distance (depth) on the image
                output_img = img_bgr.copy()

                for det in self.detections:
                    x1, y1, x2, y2 = det['bbox']
                    depth_m = det.get('depth_mean', 0)
                    valid_ratio = det.get('valid_ratio', 1.0)

                    # Choose color based on data quality
                    if valid_ratio > 0.8:
                        color = (0, 255, 0)  # Green - Excellent
                    elif valid_ratio > 0.6:
                        color = (0, 255, 255)  # Yellow - Good
                    elif valid_ratio > 0.4:
                        color = (0, 165, 255)  # Orange - Medium
                    else:
                        color = (0, 0, 255)  # Red - Poor

                    # Draw bounding box
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)

                    # ✅ Only distance in meters is written on the image
                    distance_text = f"{depth_m:.2f}m"
                    cv2.putText(output_img, distance_text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Display results in GUI
                self.result_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                self.root.after(0, lambda: self.display_image(self.result_image, self.result_label))

                # Display text results (with all details in Results panel)
                self.root.after(0, lambda: self.display_results())
                self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

            finally:
                self.root.after(0, lambda: self.detect_btn.config(state=tk.NORMAL, text="🔍 Detect Objects & Depth"))

        thread = threading.Thread(target=process)
        thread.daemon = True
        thread.start()

    def display_results(self):
        """Display results in text panel with all details"""
        self.results_text.delete(1.0, tk.END)

        if not self.detections:
            self.results_text.insert(tk.END, "No objects detected!\n")
            return

        outlier_method = self.outlier_method.get()

        self.results_text.insert(tk.END, "=" * 55 + "\n")
        self.results_text.insert(tk.END, f"📊 RESULTS ({len(self.detections)} objects)\n")
        self.results_text.insert(tk.END, "=" * 55 + "\n\n")

        for i, det in enumerate(self.detections, 1):
            self.results_text.insert(tk.END, f"{i}. {det['class']}\n")
            self.results_text.insert(tk.END, f"   📍 Confidence: {det['confidence']:.2%}\n")

            if 'depth_mean' in det:
                self.results_text.insert(tk.END, f"   📏 Distance: {det['depth_mean']:.2f} meters\n")
                self.results_text.insert(tk.END, f"   📊 Median Depth: {det['depth_median']:.2f} meters\n")

                if outlier_method != 'none' and self.show_info_var.get():
                    self.results_text.insert(tk.END, f"   ✅ Valid pixels: {det.get('valid_ratio', 0):.1%}\n")
                    self.results_text.insert(tk.END, f"   ❌ Outliers removed: {det.get('outlier_removed', 0)}\n")

                    if 'depth_min' in det and 'depth_max' in det:
                        self.results_text.insert(tk.END,
                                                 f"   📈 Range: [{det['depth_min']:.2f}, {det['depth_max']:.2f}]m\n")

                    if 'depth_std' in det:
                        self.results_text.insert(tk.END, f"   📉 Std Dev: {det['depth_std']:.3f}m\n")

            self.results_text.insert(tk.END, "\n")

        # Color explanation
        self.results_text.insert(tk.END, "-" * 55 + "\n")
        self.results_text.insert(tk.END, "🎨 Color Code (Bounding Box):\n")
        self.results_text.insert(tk.END, "   🟢 Green: Excellent (>80% valid)\n")
        self.results_text.insert(tk.END, "   🟡 Yellow: Good (60-80% valid)\n")
        self.results_text.insert(tk.END, "   🟠 Orange: Medium (40-60% valid)\n")
        self.results_text.insert(tk.END, "   🔴 Red: Poor (<40% valid)\n")

    def save_result(self):
        """Save result image"""
        if self.result_image is None:
            messagebox.showwarning("No Result", "No detection result to save")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
        )

        if file_path:
            result_bgr = cv2.cvtColor(self.result_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, result_bgr)
            messagebox.showinfo("Success", f"Result saved to:\n{file_path}")


def main():
    root = tk.Tk()
    app = DepthYOLOApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()