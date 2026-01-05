import streamlit as st
import torch
import torchaudio
import numpy as np
from PIL import Image
import io
import json

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Mumbai Bird Call Identifier",
    page_icon="🐦",
    layout="centered"
)

# ================== LOAD MODEL & LABEL MAP ==================
@st.cache_resource
def load_model_and_map():
    # Load the checkpoint
    checkpoint = torch.load("multi_species_model.pth", map_location="cpu")
    
    # Re-create the model architecture
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v3_small', pretrained=False)
    num_classes = len(checkpoint['label_map'])
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get class names (scientific names)
    class_names = list(checkpoint['label_map'].keys())
    
    return model, class_names

model, class_names = load_model_and_map()

# ================== MEL TRANSFORM (same as training) ==================
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=22050,
    n_fft=512,
    win_length=512,
    hop_length=256,
    f_min=50,
    f_max=11000,
    n_mels=128,
    norm='slaney',
    mel_scale='slaney'
)
db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
full_transform = torch.nn.Sequential(mel_transform, db_transform)

# ================== APP UI ==================
st.title("🐦 Mumbai Balcony Bird Call Identifier")
st.markdown("""
Trained on **204 Indian bird species** from real urban recordings in Mumbai & Maharashtra.  
Just record 5 seconds from your phone or laptop microphone!
""")

audio_data = st.experimental_audio_input("Click and record 5 seconds", duration=5)

if audio_data:
    st.audio(audio_data, format="audio/wav")
    
    with st.spinner("Processing audio and predicting..."):
        # Load audio from bytes
        waveform, original_sr = torchaudio.load(io.BytesIO(audio_data.getvalue()))
        
        # Resample to 22050
        if original_sr != 22050:
            waveform = torchaudio.functional.resample(waveform, original_sr, 22050)
        
        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        
        # Pad or truncate to exactly 5 seconds
        target_samples = 22050 * 5
        if waveform.shape[1] < target_samples:
            waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.shape[1]))
        else:
            waveform = waveform[:, :target_samples]
        
        # Compute Mel spectrogram
        mel = full_transform(waveform)  # (1, 128, time)
        mel = mel.squeeze(0)  # (128, time)
        
        # Normalize for visualization
        mel_min = mel.min()
        mel_max = mel.max()
        mel_norm = (mel - mel_min) / (mel_max - mel_min + 1e-8)
        
        # Prepare for model: resize to 224x224, add batch & RGB channels
        mel_input = mel.unsqueeze(0).unsqueeze(0)  # (1, 1, 128, time)
        mel_input = torch.nn.functional.interpolate(mel_input, size=(224, 224), mode='bilinear', align_corners=False)
        mel_input = mel_input.repeat(1, 3, 1, 1)  # to RGB
        
        # Inference
        with torch.no_grad():
            output = model(mel_input)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            top5_probs, top5_idx = torch.topk(probs, 5)
        
        # Display results
        st.success("Top 5 Predictions")
        for i in range(5):
            species = class_names[top5_idx[i]]
            confidence = top5_probs[i].item()
            st.markdown(f"**{species}** — {confidence:.1%}")
        
        # Show spectrogram
        mel_vis = mel_norm.cpu().numpy()
        st.image(mel_vis, caption="Your audio as Mel Spectrogram", use_column_width=True, clamp=True)

else:
    st.info("Press the microphone button above to record a 5-second bird call!")

st.markdown("---")
st.caption("Model trained on 8000+ real urban recordings from xeno-canto (Mumbai & Maharashtra area).")