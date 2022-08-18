from unicodedata import name
from main import app
import os
from flask import request, redirect, render_template
from werkzeug.utils import secure_filename
from main.recognition import takecommand
from main.cnn_predict import prediksi

audio_text = ''

@app.route('/')
def index():
    return render_template('index.html')

current_path = os.getcwd()
app.config["AUDIO_UPLOADS"] = current_path + "\\main\\static\\audio\\upload\\"
app.config["ALLOWED_AUDIO_EXTENSIONS"] = ["WAV"]

def allowed_audio(filename):
    if not "." in filename:
        return False
    ext = filename.rsplit(".", 1)[1]
    if ext.upper() in app.config["ALLOWED_AUDIO_EXTENSIONS"]:
        return True
    else:
        False


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if request.files:
            audio = request.files["audio"]
            if audio.filename == "":
                global peringatan
                peringatan = 'File Harus Diisi'
                return redirect(request.url)
            if not allowed_audio(audio.filename):
                peringatan = 'File Harus Menggunakan Format *.wav'
                return redirect(request.url)
            else:
                filename = secure_filename(audio.filename)
                audio.save(os.path.join(app.config["AUDIO_UPLOADS"], filename))
                global audio_text
                audio_text = takecommand(app.config["AUDIO_UPLOADS"] + audio.filename) # Untuk convert audio ke text
                global hasil_prediksi
                hasil_prediksi = prediksi([audio_text])
            return redirect('/hasil')
    return render_template("index.html", name=peringatan)


@app.route('/hasil')
def hasil():
    print(hasil_prediksi)
    return render_template('hasil.html', translate_text = audio_text, hasil_identifikasi = hasil_prediksi)


@app.route('/about')
def about():
    return render_template('about.html')