import os
import sys
import uuid
from flask import Flask, render_template, request
from flaskwebgui import FlaskUI
from qsi import io
import numpy as np


if __package__:
    from .. import metrics
else:
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    if ROOT_DIR not in sys.path:
        sys.path.append(ROOT_DIR)
    import metrics

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # limit to 50MB, to avoid 413 error.

def analyze(csv, save_local=False):

    # store html result into a local html file

    if save_local:

        fn = os.path.dirname(os.path.realpath(__file__)) + \
            "/" + str(uuid.uuid4()) + ".html"

        with open(fn, 'w') as f:
            f.write(analyze_file(csv))

        # fn is the local save path

    return analyze_file(csv)  # return the html content

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("home.html")


@app.route("/submit", methods=['POST'])
def run_ackl():
    r = ''
    kernel_type = request.form['kernel_type']
    use_sample = request.form["use_sample"]

    if (use_sample == 'true' or use_sample is True):
        # path = "ackl/gui/static/sample.csv"
        X, y, X_names, desc, labels = io.load_dataset('salt', x_range=list(range(400, 1000)), display=False)

    else:
        fn = os.path.dirname(os.path.realpath(
            __file__)) + "/" + str(uuid.uuid4()) + ".csv"
        request.files['dataFile'].save(fn)

        if os.path.isfile(fn) is False:
            r = 'File ' + fn + ' does not exist.'
            return {'message': 'success', 'html': r}

        X, y, X_names, labels = io.open_dataset(fn)
        print(X.shape, y.shape)

    try:
        pkl = metrics.preview_kernels(X, y,metrics=True,
                                        scatterplot=False,
                                        selected_kernel_names= [kernel_type])
        
        r = metrics.visualize_metric_dicts(pkl, plot=False)
    except Exception as e:
        r = str(e)

    return {'message': 'success', 'html': r}

if __name__ == '__main__':
    FlaskUI(app=app, server="flask", port=5052).run()
