import os
import torch
import torchaudio
import speech_recognition as sr
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinterdnd2 import TkinterDnD, DND_FILES
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def convert_mp3_to_wav(mp3_file):
    """Convert MP3 file to WAV using pydub."""
    audio = AudioSegment.from_mp3(mp3_file)
    wav_file = mp3_file.replace('.mp3', '.wav')
    audio.export(wav_file, format="wav")
    return wav_file

def transcribe_audio(file_path):
    try:

        if file_path.lower().endswith(".mp3"):
            file_path = convert_mp3_to_wav(file_path)
        
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values
        input_values = input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])
        return transcription
    except Exception as e:
        return f"Error: {str(e)}"

def transcribe_and_display():
    if hasattr(app, "file_path"):
        transcription = transcribe_audio(app.file_path)
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, transcription)
    else:
        messagebox.showwarning("No file selected", "Please select or drop a WAV or MP3 file.")

def transcribe_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, "Listening...\n")
        app.update()
        audio = recognizer.listen(source, phrase_time_limit=5)
        output_text.insert(tk.END, "Processing...\n")
        app.update()
        try:
            with open("temp_mic.wav", "wb") as f:
                f.write(audio.get_wav_data())
            result = transcribe_audio("temp_mic.wav")
            output_text.delete(1.0, tk.END)
            output_text.insert(tk.END, result)
            os.remove("temp_mic.wav")
        except Exception as e:
            output_text.insert(tk.END, f"Error: {str(e)}")

def open_file_dialog():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
    if file_path:
        file_label.config(text=os.path.basename(file_path))
        transcribe_button.config(state=tk.NORMAL)
        app.file_path = file_path

def on_drop(event):
    file_path = event.data.strip('{}')  # handle paths with spaces
    if file_path.lower().endswith((".wav", ".mp3")):
        app.file_path = file_path
        file_label.config(text=os.path.basename(file_path))
        transcribe_button.config(state=tk.NORMAL)
        transcribe_and_display()
    else:
        messagebox.showerror("Unsupported file", "Please drop a WAV or MP3 audio file.")

app = TkinterDnD.Tk()
app.title("Speech-to-Text (Wav2Vec2 English Only)")
app.geometry("540x480")
app.resizable(False, False)

select_button = tk.Button(app, text="Select Audio File", command=open_file_dialog)
select_button.pack(pady=10)

mic_button = tk.Button(app, text="Record from Microphone", command=transcribe_microphone)
mic_button.pack(pady=5)

transcribe_button = tk.Button(app, text="Transcribe Selected File", command=transcribe_and_display, state=tk.DISABLED)
transcribe_button.pack(pady=5)

file_label = tk.Label(app, text="No file selected", fg="gray")
file_label.pack()

drop_label = tk.Label(app, text="Or drag & drop an audio file below ↓", fg="blue")
drop_label.pack()

drop_zone = tk.Label(app, text="🡇 Drop Audio File Here 🡇", bg="#f0f0f0", relief="groove", height=3)
drop_zone.pack(pady=10, fill=tk.X, padx=50)
drop_zone.drop_target_register(DND_FILES)
drop_zone.dnd_bind("<<Drop>>", on_drop)

output_text = tk.Text(app, wrap=tk.WORD, height=14, width=64)
output_text.pack(pady=10)

# Run the GUI
app.mainloop()
