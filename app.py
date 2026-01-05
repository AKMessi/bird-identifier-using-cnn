import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
from torchvision import models

# Try to import torchaudio, fall back to soundfile if not available
try:
    import torchaudio
    import torchaudio.functional
    import torchaudio.transforms
    TORCHAUDIO_AVAILABLE = True
except Exception as e:
    TORCHAUDIO_AVAILABLE = False
    try:
        import soundfile as sf
    except ImportError:
        st.error("Neither torchaudio nor soundfile is available. Please check your installation.")
        st.stop()

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
    
    # Debug: Check what's in the checkpoint
    st.write("🔍 **Checkpoint Keys:**", list(checkpoint.keys()))
    
    # Get label map
    label_map = checkpoint['label_map']
    st.write(f"📋 **Number of classes in checkpoint:** {len(label_map)}")
    st.write(f"📝 **First 5 species in label_map:**", list(label_map.keys())[:5])
    st.write(f"🔢 **Label map type:**", type(label_map))
    
    # Create model directly from torchvision instead of torch.hub
    model = models.mobilenet_v3_small(pretrained=False)
    num_classes = len(label_map)
    
    st.write(f"🧠 **Model output classes:** {num_classes}")
    st.write(f"🔧 **Original classifier final layer:** {model.classifier[3]}")
    
    # Replace final layer
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
    st.write(f"✅ **New classifier final layer:** {model.classifier[3]}")
    
    # Load state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        st.success(f"✅ Model weights loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model weights: {e}")
        st.stop()
    
    model.eval()
    
    # Get class names - THIS IS CRITICAL
    # The label_map from your checkpoint should be {species_name: index}
    # We need to create a list where list[index] = species_name
    
    if isinstance(list(label_map.keys())[0], str):
        # label_map is {species_name: index}, need to invert it
        st.info("📖 Label map format: {species_name: index}")
        # Create inverse mapping: index -> species_name
        index_to_species = {v: k for k, v in label_map.items()}
        # Create ordered list by index
        class_names = [index_to_species[i] for i in range(len(label_map))]
    else:
        # label_map is {index: species_name}
        st.info("📖 Label map format: {index: species_name}")
        class_names = [label_map[i] for i in sorted(label_map.keys())]
    
    st.write(f"🐦 **Total species loaded:** {len(class_names)}")
    st.write(f"🔤 **Class names sample (indices 0-4):**")
    for i in range(min(5, len(class_names))):
        st.write(f"   Index {i}: {class_names[i]}")
    
    # Verify mapping with known species
    st.write("\n🔍 **Verification - Check these mappings:**")
    test_species = ["Accipiter badius", "Passer domesticus", "Corvus splendens"]
    for species in test_species:
        if species in label_map:
            expected_idx = label_map[species]
            actual_mapping = class_names[expected_idx]
            match = "✅" if actual_mapping == species else "❌"
            st.write(f"   {match} '{species}' should be index {expected_idx}, we have: '{actual_mapping}'")
    
    return model, class_names

model, class_names = load_model_and_map()

st.markdown("---")

# Show status of audio backend
if not TORCHAUDIO_AVAILABLE:
    st.info("ℹ️ Using soundfile backend for audio processing (torchaudio not available)")

# ================== MEL TRANSFORM (same as training) ==================
if TORCHAUDIO_AVAILABLE:
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
else:
    # Fallback mel transform using torch operations
    def compute_mel_spectrogram(waveform):
        """Compute mel spectrogram using basic torch operations"""
        # Simple STFT-based spectrogram
        window = torch.hann_window(512)
        stft = torch.stft(
            waveform.squeeze(0),
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=window,
            return_complex=True
        )
        # Power spectrogram
        power = stft.abs() ** 2
        
        # Simple mel approximation (frequency binning)
        # Create 128 mel bins by grouping frequency bins
        n_mels = 128
        freq_bins = power.shape[0]
        mel_bins = []
        bin_size = freq_bins // n_mels
        
        for i in range(n_mels):
            start = i * bin_size
            end = start + bin_size if i < n_mels - 1 else freq_bins
            mel_bins.append(power[start:end].mean(dim=0))
        
        mel_spec = torch.stack(mel_bins)
        
        # Convert to dB
        mel_db = 10 * torch.log10(mel_spec + 1e-10)
        mel_db = torch.clamp(mel_db, min=mel_db.max() - 80)
        
        return mel_db.unsqueeze(0)
    
    full_transform = compute_mel_spectrogram

# ================== APP UI ==================
st.title("🐦 Mumbai Balcony Bird Call Identifier")
st.markdown("""
Trained on **204 Indian bird species** from real urban recordings in Mumbai & Maharashtra.  
Upload a 5-second bird call audio file (WAV format recommended).
""")

st.info("📱 **How to use**: Record a bird call on your phone, then upload the audio file here!")

audio_data = st.file_uploader("Upload your bird call audio (WAV, MP3, M4A)", type=["wav", "mp3", "m4a", "ogg"])

if audio_data:
    st.audio(audio_data, format="audio/wav")
    
    with st.spinner("🔄 Processing audio and predicting..."):
        try:
            # Load audio from bytes
            audio_bytes = audio_data.read()
            audio_data.seek(0)  # Reset file pointer
            
            # Debug: Show file info
            st.info(f"📁 File size: {len(audio_bytes) / 1024:.1f} KB")
            
            if TORCHAUDIO_AVAILABLE:
                try:
                    waveform, original_sr = torchaudio.load(io.BytesIO(audio_bytes))
                except Exception as e:
                    # If torchaudio fails, fall back to soundfile
                    st.warning(f"TorchAudio failed, using soundfile fallback")
                    data, original_sr = sf.read(io.BytesIO(audio_bytes))
                    waveform = torch.FloatTensor(data)
                    if len(waveform.shape) == 2:  # If stereo
                        waveform = waveform.mean(dim=1)
                    waveform = waveform.unsqueeze(0)
            else:
                # Use soundfile as fallback
                data, original_sr = sf.read(io.BytesIO(audio_bytes))
                waveform = torch.FloatTensor(data)
                if len(waveform.shape) == 2:  # If stereo
                    waveform = waveform.mean(dim=1)
                waveform = waveform.unsqueeze(0)
            
            # Debug info
            st.info(f"🎵 Original sample rate: {original_sr} Hz, Duration: {waveform.shape[1] / original_sr:.2f} seconds")
            st.info(f"📊 Waveform shape: {waveform.shape}")
            
            # Resample to 22050 if needed
            if original_sr != 22050:
                if TORCHAUDIO_AVAILABLE:
                    waveform = torchaudio.functional.resample(waveform, original_sr, 22050)
                else:
                    # Simple resampling
                    duration = waveform.shape[1] / original_sr
                    target_length = int(duration * 22050)
                    indices = torch.linspace(0, waveform.shape[1] - 1, target_length)
                    waveform = torch.nn.functional.interpolate(
                        waveform.unsqueeze(1), 
                        size=target_length, 
                        mode='linear', 
                        align_corners=True
                    ).squeeze(1)
            
            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(0, keepdim=True)
            
            # Pad or truncate to exactly 5 seconds
            target_samples = 22050 * 5
            if waveform.shape[1] < target_samples:
                waveform = torch.nn.functional.pad(waveform, (0, target_samples - waveform.shape[1]))
            else:
                waveform = waveform[:, :target_samples]
            
            st.info(f"✂️ Processed to 5 seconds: {waveform.shape}")
            
            # Compute Mel spectrogram
            if TORCHAUDIO_AVAILABLE:
                mel = full_transform(waveform)  # (1, 128, time)
                mel = mel.squeeze(0)  # (128, time)
            else:
                mel = full_transform(waveform)  # (1, 128, time)
                mel = mel.squeeze(0)  # (128, time)
            
            st.info(f"🎼 Mel spectrogram shape: {mel.shape}")
            
            # Check if mel spectrogram is valid
            if torch.isnan(mel).any() or torch.isinf(mel).any():
                st.error("⚠️ Invalid mel spectrogram detected (NaN or Inf values)")
                st.stop()
            
            # Show mel spectrogram statistics
            st.info(f"📈 Mel stats - Min: {mel.min():.2f}, Max: {mel.max():.2f}, Mean: {mel.mean():.2f}")
            
            # Normalize for visualization
            mel_min = mel.min()
            mel_max = mel.max()
            mel_norm = (mel - mel_min) / (mel_max - mel_min + 1e-8)
            
            # Prepare for model: resize to 224x224, add batch & RGB channels
            mel_input = mel.unsqueeze(0).unsqueeze(0)  # (1, 1, 128, time)
            st.info(f"🔧 Before resize: {mel_input.shape}")
            
            mel_input = torch.nn.functional.interpolate(mel_input, size=(224, 224), mode='bilinear', align_corners=False)
            st.info(f"📐 After resize to 224x224: {mel_input.shape}")
            
            mel_input = mel_input.repeat(1, 3, 1, 1)  # to RGB
            st.info(f"🎨 After RGB conversion: {mel_input.shape}")
            
            # Show input statistics
            st.info(f"🔢 Model input stats - Min: {mel_input.min():.2f}, Max: {mel_input.max():.2f}, Mean: {mel_input.mean():.2f}")
            
            # Inference
            with torch.no_grad():
                output = model(mel_input)
                st.info(f"🧠 Raw model output shape: {output.shape}")
                st.info(f"📊 Raw output stats - Min: {output.min():.2f}, Max: {output.max():.2f}")
                
                probs = torch.nn.functional.softmax(output[0], dim=0)
                st.info(f"🎲 Probabilities sum: {probs.sum():.4f} (should be ~1.0)")
                
                top5_probs, top5_idx = torch.topk(probs, 5)
            
            # Show raw top 5 for debugging
            with st.expander("🔍 DEBUG: Raw Top 5 Predictions"):
                for i in range(5):
                    st.write(f"{i+1}. Index: {top5_idx[i].item()}, Prob: {top5_probs[i].item():.4f}, Species: {class_names[top5_idx[i]]}")
            
            # Determine confidence level
            top1_confidence = top5_probs[0].item()
            top1_species = class_names[top5_idx[0]]
            
            # Confidence thresholds
            HIGH_CONFIDENCE = 0.60  # 60% or higher = very confident
            MEDIUM_CONFIDENCE = 0.35  # 35-60% = somewhat confident
            
            # Display primary result
            st.markdown("---")
            if top1_confidence >= HIGH_CONFIDENCE:
                # High confidence - show single prediction
                st.success("✅ **Bird Identified with High Confidence**")
                st.markdown(f"### 🐦 {top1_species}")
                st.markdown(f"**Confidence:** {top1_confidence:.1%}")
                st.progress(top1_confidence)
                
                st.markdown("---")
                st.markdown("**Alternative possibilities:**")
                for i in range(1, min(3, len(top5_probs))):
                    species = class_names[top5_idx[i]]
                    confidence = top5_probs[i].item()
                    st.markdown(f"- {species} ({confidence:.1%})")
                    
            elif top1_confidence >= MEDIUM_CONFIDENCE:
                # Medium confidence - show top prediction with note
                st.warning("⚠️ **Likely Match (Medium Confidence)**")
                st.markdown(f"### 🐦 Most Likely: {top1_species}")
                st.markdown(f"**Confidence:** {top1_confidence:.1%}")
                st.progress(top1_confidence)
                
                st.info("The model is somewhat confident but not certain. Check the alternatives below:")
                st.markdown("**Other strong possibilities:**")
                for i in range(1, min(4, len(top5_probs))):
                    species = class_names[top5_idx[i]]
                    confidence = top5_probs[i].item()
                    st.markdown(f"**{i+1}.** {species} — {confidence:.1%}")
                    
            else:
                # Low confidence - uncertain/unknown
                st.error("❓ **Unable to Identify with Confidence**")
                st.markdown("### Unknown / Other Species")
                st.markdown(f"Top match confidence is only **{top1_confidence:.1%}**, which is too low for reliable identification.")
                
                with st.expander("🔍 Show top 5 possibilities (all low confidence)"):
                    st.warning("⚠️ All predictions below have low confidence. The recording may contain:")
                    st.markdown("""
                    - A species not in our training dataset
                    - Poor audio quality or too much background noise
                    - Multiple birds calling simultaneously
                    - Non-bird sounds
                    """)
                    
                    for i in range(5):
                        species = class_names[top5_idx[i]]
                        confidence = top5_probs[i].item()
                        st.markdown(f"**{i+1}.** {species} — {confidence:.1%}")
                
                st.info("💡 **Suggestions:** Try recording again with less background noise, get closer to the bird, or record during active calling periods (early morning/evening).")
            
            # Show spectrogram
            st.markdown("---")
            with st.expander("📊 View Audio Spectrogram"):
                mel_vis = mel_norm.cpu().numpy()
                st.image(mel_vis, caption="Mel Spectrogram of your audio", use_container_width=True, clamp=True)
                st.caption("This visualization shows the frequency content of the bird call over time.")
            
        except Exception as e:
            st.error(f"❌ Error processing audio: {str(e)}")
            st.info("Please try with a different audio file or check the file format.")
            with st.expander("🔧 Troubleshooting"):
                st.markdown("""
                **Common issues:**
                - File format not supported (use WAV, MP3, or M4A)
                - Audio file is corrupted
                - Recording is too short or too long
                - File size too large
                
                **Tips:**
                - Recommended: WAV format, 5-10 seconds
                - Keep file size under 10MB
                - Ensure clear audio with minimal noise
                """)

else:
    st.info("👆 Upload a bird call audio file to get started!")
    
    # Show model info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Species Coverage", "204 birds")
    with col2:
        st.metric("Training Data", "8000+ calls")
    with col3:
        st.metric("Region", "Mumbai & MH")
    
    with st.expander("ℹ️ Tips for best results"):
        st.markdown("""
        **Recording Tips:**
        - Record in a quiet location (early morning or evening is best)
        - Get as close to the bird as safely possible
        - Record for at least 5 seconds
        - Minimize background noise (traffic, wind, other birds)
        
        **Supported Formats:**
        - WAV (recommended)
        - MP3
        - M4A
        - OGG
        
        **How to Record:**
        1. Use your phone's voice recorder app
        2. Record the bird call for 5+ seconds
        3. Upload the file here
        
        **Understanding Results:**
        - **High Confidence (≥60%)**: Very likely correct identification
        - **Medium Confidence (35-60%)**: Possible match, verify with field guides
        - **Low Confidence (<35%)**: Unable to identify, marked as "Unknown"
        """)
    
    with st.expander("🔍 What species can this identify?"):
        st.markdown("""
        This model recognizes **204 Indian bird species** commonly found in Mumbai and Maharashtra, including:
        
        **Common Urban Birds:**
        - House Sparrow, Common Myna, House Crow
        - Red-vented Bulbul, Asian Koel
        - Spotted Dove, Rock Pigeon
        
        **Special Species:**
        - Indian Pitta, Forest Owlet (Critically Endangered)
        - Grey Junglefowl, Malabar Whistling Thrush
        - Various warblers, flycatchers, and babblers
        
        **Note:** Species not in our training dataset will show as "Unknown/Other"
        """)

st.markdown("---")
st.caption("Model trained on 8000+ real urban recordings from xeno-canto (Mumbai & Maharashtra area).")
st.caption("⚠️ AI predictions should be verified with expert knowledge for critical identifications.")