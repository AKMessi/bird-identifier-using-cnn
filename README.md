# 🐦 Mumbai Bird Call Identifier

An AI-powered bird call identification system trained specifically on **204 Indian bird species** commonly found in **Mumbai and Maharashtra**.  
This application allows you to record bird calls directly from your device and get **instant species predictions**.

---

## 🎯 Features

- **Real-time Audio Recording**  
  Record 5-second bird calls directly in your browser

- **Instant Classification**  
  Get **top-5 species predictions** with confidence scores

- **Urban-Focused Dataset**  
  Trained on **8000+ real recordings** from Mumbai & Maharashtra

- **Visual Feedback**  
  View the **mel spectrogram** of your recording

- **204 Species Coverage**  
  From common urban birds to rare endemic species

---

## 🚀 Try It Out

Simply click the **microphone button**, record a **5-second bird call**, and get instant predictions!

---

## 🔬 Model Details

### Architecture

- **Base Model:** MobileNetV3-Small (PyTorch)  
- **Input:** Mel Spectrogram (128 mel bins, 224 × 224 pixels)

#### Audio Preprocessing

- Sample Rate: `22,050 Hz`
- FFT Size: `512`
- Hop Length: `256`
- Frequency Range: `50 Hz – 11,000 Hz`
- Normalization: Slaney norm with amplitude → dB conversion

---

## 🧠 Training Data

- **Source:** Xeno-canto (citizen science bird sound database)
- **Geographic Focus:** Mumbai & Maharashtra, India
- **Total Recordings:** 8000+ urban & suburban clips
- **Species Count:** 204 Indian bird species
- **Clip Duration:** 5 seconds per recording

---

## 🐤 Species Coverage

The model recognizes a wide range of Indian birds, including:

### Common Urban Birds

- House Sparrow (*Passer domesticus*)
- Common Myna (*Acridotheres tristis*)
- House Crow (*Corvus splendens*)
- Red-vented Bulbul (*Pycnonotus cafer*)
- Asian Koel (*Eudynamys scolopaceus*)

### Endemic & Special Species

- Indian Pitta (*Pitta brachyura*)
- Forest Owlet (*Athene blewitti*) — **Critically endangered**
- Grey Junglefowl (*Gallus sonneratii*)
- Malabar Whistling Thrush (*Myophonus horsfieldii*)

📄 **Complete Species List:** See `label_map.json` for all 204 species.

---

## 📊 Performance

### Performs Best When:

- Recordings are clear with minimal background noise
- Audio conditions match urban/suburban environments
- Species have distinctive and well-represented calls

### Performance May Vary For:

- Rare species with fewer training samples
- Noisy environments
- Species with similar-sounding vocalizations

---

## 🛠️ Technical Implementation

### Audio Processing Pipeline

