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
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # limit to 5MB

def load_file(pathname):
    '''
    Load data from a csv file
    '''
    M = np.loadtxt(pathname, delimiter=',', skiprows=1)
    X = M[:, :-1]
    y = M[:, -1].astype(int)
    return X, y

def analyze_file(fn):
    if os.path.isfile(fn) == False:
        return 'File ' + fn + ' does not exist.'

    X, y = load_file(fn)
    return X, y

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


@app.route("/submit", methods=['GET', 'POST'])
def run_ackl():    
    if request.method == 'POST':

        r = ''
        kernel_type = request.form['name']
        print(kernel_type)
        use_sample = request.form["use_sample"]
        print(use_sample)

        if (use_sample):
            # path = "ackl/gui/static/sample.csv"
            X, y, X_names, desc, labels = io.load_dataset('salt', x_range=list(range(400, 1000)), display=False)
            pkl = metrics.preview_kernels(X, y, metrics=True,
                                          scatterplot=False,
                                          selected_kernel_names= [kernel_type])
            r = metrics.visualize_metric_dicts(pkl, plot=False)
        else:
            f = request.files['dataFile']
            csv = os.path.dirname(os.path.realpath(
                __file__)) + "/" + str(uuid.uuid4()) + ".csv"
            f.save(csv)
            print('1')
            X,y = analyze_file(csv)
            pkl = metrics.preview_kernels(X, y,metrics=True,
                                          scatterplot=False,
                                          selected_kernel_names= [kernel_type])
            r = metrics.visualize_metric_dicts(pkl, plot=False)


    return {'message': 'success', 'html': r}
    #render_template("home.html", use_sample = use_sample,  cla_result = r)

# def open_browser():
#    webbrowser.open_new('http://localhost:5052/')

if __name__ == '__main__':
    # # use netstat -ano|findstr 5051 to check port use
    # Timer(3, open_browser).start()
    # app.run(host="0.0.0.0", port=5052, debug=False)
    FlaskUI(app=app, server="flask", port=5052).run()