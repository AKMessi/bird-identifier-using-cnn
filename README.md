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

Raw Audio (WAV)
→ Resample to 22.05 kHz
→ Convert to Mono
→ Pad / Truncate to 5 seconds
→ Mel Spectrogram
→ dB Normalization
→ Resize to 224 × 224
→ Model Inference

shell
Copy code

### Model Architecture

MobileNetV3-Small
├── Feature Extractor (Pretrained)
└── Custom Classifier
└── Linear Layer (204 classes)

yaml
Copy code

---

## 📁 Repository Structure

.
├── app.py # Streamlit application
├── multi_species_model.pth # Trained model checkpoint
├── requirements.txt # Python dependencies
├── label_map.json # Species name → index mapping
└── README.md # Documentation

yaml
Copy code

---

## 🔧 Local Setup

### Prerequisites

- Python **3.8 – 3.11**  
  *(Python 3.13 not yet supported by PyTorch)*
- **2GB+ RAM** recommended

### Installation

```bash
# Clone the repository
git clone https://huggingface.co/spaces/AKMESSI/bird-identifier
cd bird-identifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
The app will open in your browser at:
👉 http://localhost:8501

📝 Usage Tips
For Best Results
Recording Environment
Choose a quiet location

Record early morning or evening

Get close to the bird safely

Recording Quality
Use a good-quality microphone

Avoid wind noise

Minimize background sounds

Species Identification
Review all top-5 predictions

Cross-check with visual ID

Consider habitat & region

📈 Understanding Results
High Confidence (>70%)
Likely correct identification

Medium Confidence (30–70%)
Possible match — verify manually

Low Confidence (<30%)
Uncertain — improve recording quality

🌍 Species Distribution
Passerines (Perching birds): ~60%

Raptors (Birds of prey): ~8%

Waterbirds: ~12%

Owls & Nightjars: ~6%

Others (Parrots, Woodpeckers, etc.): ~14%

⚠️ Limitations
Geographic bias (Mumbai/Maharashtra optimized)

Sensitive to background noise

Seasonal call variations

Confusion between similar species

Not a replacement for expert verification

🎓 Educational Use
Designed for:

Bird watchers

Citizen science projects

Education

Wildlife documentation

Ecological research

Not recommended for professional ornithological research without expert validation.

📚 Data Sources & Acknowledgments
Audio Data: Xeno-canto

Contributors: Thousands of citizen scientists

Taxonomy: IOC World Bird List

Special thanks to the Xeno-canto community for making this project possible.

🤝 Contributing
Contributions are welcome!

Areas to improve:

More species coverage

Higher accuracy

Common-name support

Multi-language support

Noise robustness

Mobile app development

📄 License
Licensed under the MIT License.
See the LICENSE file for details.

⚠️ Audio data from Xeno-canto is under Creative Commons licenses.
Check individual recordings on xeno-canto.org.

🔗 Links
Model Repository: Hugging Face Space

Issue Tracker: GitHub Issues

Xeno-canto: https://xeno-canto.org

📧 Contact
For questions, suggestions, or collaborations, please open an issue on the repository.

Disclaimer:
This is an AI-based tool. Predictions should always be verified, especially for conservation or research purposes. Consult ornithological experts for critical identifications.