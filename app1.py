# ====================================================
# 🧠 Brain MRI Analyzer — Full Pipeline Streamlit App
# Stage 1: Healthy vs Diseased (Hugging Face ViT)
# Stage 2: MCI vs Alzheimer (Custom ViT from timm)
# Stage 3: UNet Segmentation + Region Analysis + 3D Overlay
# ====================================================

import streamlit as st
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from nilearn import datasets, plotting
from nilearn.image import load_img, resample_to_img
from transformers import ViTForImageClassification
from timm import create_model
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from datetime import datetime
import os, tempfile, shutil

# ====================================================
# 🌸 STREAMLIT CONFIG
# ====================================================
st.set_page_config(page_title="🧠 Brain MRI Analyzer", layout="wide")
st.markdown("""
<style>
body {background-color:#f9fbfc;font-family:'Poppins',sans-serif;color:#1c1c1c;}
h1,h2,h3{color:#004b63;font-weight:600;}
.healthy{font-size:2em;color:#1ea97c;font-weight:800;}
.diseased{font-size:2em;color:#e63946;font-weight:800;}
.stTabs [role="tablist"] button {
    background-color:#eaf6f9;color:#004b63;border-radius:10px;
    margin-right:8px;font-weight:600;
}
.stTabs [aria-selected="true"] {
    background-color:#004b63 !important;color:white !important;
}
</style>
""", unsafe_allow_html=True)
st.title("🧩 MRI SEGMENTATION FOR BRAIN DISEASE PREDICTION")

progress = st.sidebar.progress(0, text="Initializing...")

# ====================================================
# 🧠 MODEL LOADERS
# ====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_stage1_model():
    model_path = r"C:\Users\Aadya\OneDrive\Desktop\Brain\vit_output\checkpoint-6"
    model = ViTForImageClassification.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_stage2_model():
    model_path = r"C:\Users\Aadya\OneDrive\Desktop\Brain\vit_multiclass_model\vit_model.pth"
    model = create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
    model.head = nn.Linear(model.head.in_features, 2)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

vit_stage1 = load_stage1_model()
progress.progress(30, text="✅ Stage 1 Model Loaded")

vit_stage2 = load_stage2_model()
progress.progress(60, text="✅ Stage 2 Model Loaded")

# ====================================================
# 🧬 UNET MODEL
# ====================================================
class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv3D(in_channels, 32)
        self.pool = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3D(32, 64)
        self.enc3 = DoubleConv3D(64, 128)
        self.bottleneck = DoubleConv3D(128, 256)
        self.up3 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv3D(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv3D(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv3D(64, 32)
        self.final_conv = nn.Conv3d(32, out_channels, 1)
    def forward(self, x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1)); e3=self.enc3(self.pool(e2))
        b=self.bottleneck(self.pool(e3))
        d3=self.dec3(torch.cat([self.up3(b),e3],dim=1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],dim=1))
        d1=self.dec1(torch.cat([self.up1(d2),e1],dim=1))
        return self.final_conv(d1)

@st.cache_resource
def load_unet():
    model = UNet3D().to(DEVICE)
    state = torch.load(r"C:\Users\Aadya\OneDrive\Desktop\Brain\unet3d_brain_binary.pth", map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

unet_model = load_unet()
progress.progress(80, text="✅ UNet Model Loaded")

# ====================================================
# 🧩 HELPERS
# ====================================================
transform_vit = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_middle_slice(nii_path):
    nii = nib.load(nii_path)
    data = nii.get_fdata()
    mid = data.shape[2] // 2
    img = (data[:, :, mid] - data.min()) / (data.max() - data.min() + 1e-8)
    return Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")

def classify_mri(nii_path, stage1, stage2):
    image = load_middle_slice(nii_path)
    tensor = transform_vit(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out1 = stage1(tensor)
        pred1 = torch.argmax(out1.logits, dim=1).item()
    stage1_label = ["Healthy", "Diseased"][pred1]
    if stage1_label == "Healthy":
        return stage1_label, None
    else:
        with torch.no_grad():
            out2 = stage2(tensor)
            pred2 = torch.argmax(out2, dim=1).item()
        stage2_label = ["MCI", "Alzheimer"][pred2]
        return stage1_label, stage2_label

def run_segmentation(model, nii_path):
    nii = nib.load(nii_path)
    data = nii.get_fdata().astype(np.float32)
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        mask = (torch.sigmoid(out).cpu().numpy().squeeze() > 0.5).astype(np.uint8)
    return nii, data, mask

def plot_middle(image, mask):
    mid = image.shape[2] // 2
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].imshow(image[:, :, mid], cmap="gray"); ax[0].set_title("Original")
    ax[1].imshow(mask[:, :, mid], cmap="gray"); ax[1].set_title("Predicted Mask")
    ax[2].imshow(image[:, :, mid], cmap="gray"); ax[2].imshow(mask[:, :, mid], cmap="Reds", alpha=0.4)
    ax[2].set_title("Overlay")
    for a in ax: a.axis("off")
    st.pyplot(fig); plt.close(fig)

# ====================================================
# 📂 UPLOAD
# ====================================================
uploaded = st.file_uploader("📤 Upload MRI (.nii / .nii.gz)", type=["nii", "nii.gz"])
if not uploaded:
    st.info("Upload an MRI file to begin.")
    st.stop()

tmp_dir = tempfile.mkdtemp()
nii_path = os.path.join(tmp_dir, uploaded.name)
with open(nii_path, "wb") as f:
    shutil.copyfileobj(uploaded, f)

progress.progress(100, text="✅ Ready")

# ====================================================
# 🧭 TABS
# ====================================================
tabs = st.tabs(["🧠 Classification", "🩸 Segmentation", "📊 Region Analysis", "📑 Report"])

# ---- TAB 1 ----
with tabs[0]:
    st.header("🧠 Two-Stage Classification")
    label1, label2 = classify_mri(nii_path, vit_stage1, vit_stage2)
    st.image(load_middle_slice(nii_path), caption="Middle Slice", width=300)
    if label1 == "Healthy":
        st.markdown(f"<p class='healthy'>✅ Stage 1: {label1}</p>", unsafe_allow_html=True)
        st.info("Brain appears healthy — skipping Stage 2 & segmentation.")
    else:
        st.markdown(f"<p class='diseased'>⚠️ Stage 1: {label1}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='diseased'>🧬 Classified as: {label2}</p>", unsafe_allow_html=True)

# ---- TAB 2 ----
with tabs[1]:
    st.header("🩸 3D UNet Segmentation")
    if label1 == "Healthy":
        st.info("Segmentation not needed for healthy brain.")
        mask = None
    else:
        nii, data, mask = run_segmentation(unet_model, nii_path)
        mask_path = os.path.join(tmp_dir, "pred_mask.nii.gz")
        nib.save(nib.Nifti1Image(mask.astype(np.float32), nii.affine), mask_path)
        plot_middle(data, mask)
        st.success("✅ Segmentation Complete!")

# ---- TAB 3 ----
with tabs[2]:
    st.header("📊 Region Analysis & 3D Visualization")
    if label1 != "Healthy":
        healthy_ref = r"C:\\Users\\Aadya\\OneDrive\\Desktop\\Brain\\datasets\\healthy_brain\\mni_icbm152_t1_tal_nlin_asym_09a.nii.gz"
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        atlas_img = load_img(atlas.maps)
        atlas_resampled = resample_to_img(atlas_img, mask_path, interpolation='nearest')
        atlas_data = atlas_resampled.get_fdata().astype(int)
        mask_data = load_img(mask_path).get_fdata() > 0
        labels = list(atlas.labels)
        region_vox = {}
        for i, name in enumerate(labels):
            if i == 0: continue
            region_mask = (atlas_data == i)
            affected = np.sum(mask_data[region_mask])
            total = np.sum(region_mask)
            if total > 0 and affected > 0:
                region_vox[name] = [affected, (affected / total) * 100]
        if region_vox:
            df = pd.DataFrame(region_vox, index=["Voxels", "Percent (%)"]).T
            df = df.sort_values(by="Percent (%)", ascending=False)
            st.dataframe(df.head(15).style.background_gradient(cmap="Reds"), use_container_width=True)

            # 🌈 BAR CHART - Top 10 Affected Regions
            st.subheader("📈 Top 10 Affected Regions")
            top10 = df.head(10)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(top10.index[::-1], top10["Percent (%)"][::-1])
            ax.set_xlabel("Percent of Region Affected (%)")
            ax.set_ylabel("Brain Region")
            ax.set_title("Top 10 Most Affected Brain Regions")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # 🧩 3D Overlay
            st.subheader("🧩 3D Overlay Visualization")
            top = df.head(10)
            top_mask = np.zeros_like(atlas_data, dtype=float)
            for name, perc in zip(top.index, top["Percent (%)"]):
                try:
                    idx = labels.index(name)
                    top_mask[atlas_data == idx] = perc
                except ValueError:
                    continue
            overlay_img = nib.Nifti1Image(top_mask, atlas_resampled.affine)
            overlay_path = os.path.join(tmp_dir, "overlay.png")
            plotting.plot_stat_map(overlay_img, bg_img=healthy_ref,
                                   title="Top 10 Affected Regions (3D Overlay)",
                                   threshold=0, display_mode='ortho',
                                   cmap='hot', output_file=overlay_path)
            st.image(overlay_path, caption="3D Overlay of Affected Regions", use_container_width=True)
        else:
            st.warning("No affected regions found.")
    else:
        st.info("Region analysis not applicable for healthy brain.")

# ---- TAB 4 ----
with tabs[3]:
    st.header("📑 Diagnostic Report")
    report_name = f"MRI_Report_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(tmp_dir, report_name)
    c = canvas.Canvas(pdf_path, pagesize=A4)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(150, 800, "🧠 Brain MRI Diagnostic Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, 750, f"Stage 1: {label1}")
    if label1 != "Healthy":
        c.drawString(50, 730, f"Stage 2: {label2}")
    c.save()
    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download Diagnostic Report", f, report_name, mime="application/pdf")

st.sidebar.success("✅ All models loaded and ready!")
