import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

# --- Konfigurasi Awal (Harus SAMA dengan saat training) ---
IMG_HEIGHT = 128
IMG_WIDTH = 800
NUM_CLASSES = 4
MODEL_PATH = "severstal_unet_pytorch_best_augmented.pth"
DEVICE = "cpu"

# --- 1. Arsitektur Model U-Net (Tidak ada perubahan) ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        return torch.sigmoid(self.final_conv(x))

# --- 2. Fungsi Helper & Preprocessing (Tidak ada perubahan) ---

@st.cache_resource
def load_model(model_path):
    model = UNET(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    model.eval()
    return model

def preprocess_image(image_pil):
    image_np = np.array(image_pil.convert('L'))
    resized_img = cv2.resize(image_np, (IMG_WIDTH, IMG_HEIGHT))
    image_tensor = torch.from_numpy(resized_img).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    return image_tensor.to(DEVICE)

def format_and_overlay_mask(image_np, mask_np):
    # Resize gambar asli agar sesuai dengan dimensi output untuk overlay yang akurat
    image_resized_for_overlay = cv2.resize(image_np, (IMG_WIDTH, IMG_HEIGHT))
    image_color = cv2.cvtColor(image_resized_for_overlay, cv2.COLOR_GRAY2BGR)
    overlay = image_color.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)] # Merah, Hijau, Biru, Kuning (RGB)

    for c in range(NUM_CLASSES):
        mask_channel = (mask_np[:, :, c] * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_channel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Konversi warna dari RGB ke BGR untuk OpenCV
        bgr_color = (colors[c][2], colors[c][1], colors[c][0])
        cv2.drawContours(overlay, contours, -1, bgr_color, 2)
    
    result = cv2.addWeighted(image_color, 0.6, overlay, 0.4, 0)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# --- 3. Antarmuka Streamlit yang Ditingkatkan ---

def main():
    st.set_page_config(page_title="Deteksi Cacat Baja", page_icon="🔩", layout="wide")

    # --- Sidebar ---
    with st.sidebar:
        st.title("🔩 Deteksi Cacat Baja")
        st.image("https://i.imgur.com/CVQ8J47.png", width=150) # Contoh logo
        st.header("Informasi Kelas Cacat")
        
        colors = ["Red", "Green", "Blue", "Yellow"]
        class_labels = ["Kelas 1", "Kelas 2", "Kelas 3", "Kelas 4"]
        
        for i, (label, color) in enumerate(zip(class_labels, colors)):
            st.markdown(f"<p style='color:{color};'>■ <b>{label}</b></p>", unsafe_allow_html=True)
            
        st.divider()
        st.info("Aplikasi ini dibuat untuk mendeteksi cacat pada permukaan baja menggunakan model U-Net.")

    # --- Halaman Utama ---
    st.title("Analisis Segmentasi Cacat pada Permukaan Baja")
    st.markdown("Unggah gambar penampang baja (format JPG, PNG, atau JPEG) untuk memulai analisis. Model akan secara otomatis mendeteksi dan melokalisasi area cacat.")
    
    # --- Area Upload ---
    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Pilih file gambar...", 
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )

    # --- Logika Utama setelah file diunggah ---
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        original_image_np = np.array(image_pil.convert('L'))
        
        # Tampilkan spinner saat model sedang bekerja
        with st.spinner("🧠 Menganalisis gambar dan melakukan segmentasi..."):
            model = load_model(MODEL_PATH)
            input_tensor = preprocess_image(image_pil)
            
            with torch.no_grad():
                pred_mask = model(input_tensor)
                pred_mask = (pred_mask > 0.5).float().cpu()

            pred_mask_squeezed = pred_mask.squeeze(0) 
            pred_mask_np = pred_mask_squeezed.permute(1, 2, 0).numpy()
            
            result_image = format_and_overlay_mask(original_image_np, pred_mask_np)

        st.divider()
        st.subheader("📊 Hasil Analisis")

        # --- Tampilan Hasil Berdampingan ---
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_pil, caption="🖼️ Gambar Asli", use_column_width=True)
        with col2:
            st.image(result_image, caption="🎨 Hasil Segmentasi Model", use_column_width=True)

        # --- Kesimpulan Deteksi ---
        st.subheader("📝 Ringkasan Deteksi")
        detected_classes = []
        class_colors_map = {0: "Red", 1: "Green", 2: "Blue", 3: "Yellow"}
        
        for i in range(NUM_CLASSES):
            # Cek apakah ada piksel cacat yang terdeteksi untuk kelas ini
            if pred_mask_np[:, :, i].sum() > 0:
                detected_classes.append(i)
        
        if not detected_classes:
            st.success("✅ **Analisis Selesai:** Tidak ada cacat yang terdeteksi pada gambar ini.")
        else:
            st.warning(f"⚠️ **Analisis Selesai:** Terdeteksi {len(detected_classes)} jenis cacat.")
            report = ""
            for i in detected_classes:
                report += f"- <p style='display:inline-block; color:{class_colors_map[i]};'><b>Cacat Kelas {i+1}</b></p> terdeteksi.\n"
            st.markdown(report, unsafe_allow_html=True)

    else:
        st.info("ℹ️ Menunggu gambar untuk dianalisis. Silakan unggah file di atas.")

    # --- Expander untuk Detail Teknis ---
    st.divider()
    with st.expander("ℹ️ Lihat Detail Teknis Model"):
        st.write("""
        - **Model:** U-Net
        - **Framework:** PyTorch
        - **Dimensi Input:** Gambar diubah ukurannya menjadi `128x800` piksel sebelum diproses.
        - **Output:** Mask segmentasi dengan 4 channel, masing-masing merepresentasikan satu kelas cacat.
        - **Deployment:** Dijalankan menggunakan Streamlit.
        - **Waktu & Lokasi Saat Ini:** Rabu, 11 Juni 2025, Surabaya, Indonesia.
        """)


if __name__ == "__main__":
    main()
