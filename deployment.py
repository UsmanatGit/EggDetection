import os
os.environ['KMP_DUPLICATE_LIB_OK'] ='TRUE'
# Required Libraries
from flask import Flask, Response, session, render_template, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, StringField, DecimalField, IntegerField
from wtforms.validators import InputRequired, number_range
from werkzeug.utils import secure_filename

import os
import cv2
from main import egg_detect

app = Flask(__name__)

app.config['SECRET_KEY'] = 'usman'
app.config['UPLOAD_FOLDER'] = 'static/videos'

# using flaskforms to get input video from user

class UploadVideoForm(FlaskForm):
    file = FileField('Video', validators=[InputRequired()])
    submit = SubmitField("Run")


# displaying the outuput with detection and counts
def generate_result(file=''):

    yolo_output = egg_detect(file)

    for output in yolo_output:

        ref, buffer = cv2.imencode('.jpg', output)

        frame = buffer.tobytes()
        yield  (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')

@app.route('/', methods=['GET','POST'])
@app.route('/home', methods=['GET','POST'])

def home():
    session.clear()
    return render_template('index.html')

# 'app.route()' method, to render the Webcam page at "/webcam"
@app.route("/webcam", methods=['GET','POST'])
def webcam():
    session.clear()
    return render_template('realtime.html')

@app.route('/upload', methods=['GET','POST'])
def uplaod():
    # Upload File Form: Create an instance for the Upload File Form
    form = UploadVideoForm()
    if form.validate_on_submit():
        # Our uploaded video file path is saved here
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))  # Then save the file

        # Use session storage to save video file path
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('upload_video.html', form=form)

@app.route('/upload_video')
def upload_video():
    return Response(generate_result(session.get('video_path', None)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_webcam')
def live_webcam():
    return Response(generate_result(file=0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_streaming', methods=['POST'])
def stop_streaming():
    global stop_streaming
    stop_streaming = True
    return redirect(url_for('home'))


if __name__ == "__main__":
    app.run(debug=True)