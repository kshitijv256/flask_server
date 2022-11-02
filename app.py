import os
from flask import Flask, flash, request, redirect, url_for
from flask import send_from_directory
from werkzeug.utils import secure_filename

import solution

##
with open('views/main.html', 'r') as f:
    mainPage = f.read()
with open('views/results.html', 'r') as f:
    resultsPage = f.read()
##

UPLOAD_FOLDER = 'database/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','java'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('results', name=filename))
    return mainPage

@app.route('/results/<name>')
def results(name):
    # return send_from_directory(app.config['UPLOAD_FOLDER'], name)

    labelEncoder = solution.prepareEncoder()
    model = solution.loadModel()
    img = solution.loadImg(os.path.join(app.config['UPLOAD_FOLDER'], name))
    detectedList = solution.findContours(img)
    pred,result = solution.predictResult(model, detectedList, labelEncoder)

    if result is None:
        st = "Unable to solve: "+pred
        return resultsPage.replace('replaceit', st)
    else:
        st = pred+' = '+str(result)
        return resultsPage.replace('replaceit', st)


app.add_url_rule('/results/<name>', 'results', build_only=True)

if __name__ == "__main__":
    app.run()