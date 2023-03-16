import os
import sys
import uuid
from flask import Flask, render_template, request
from flaskwebgui import FlaskUI
from qsi import io

if __package__:
    from .. import metrics
else:
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    if ROOT_DIR not in sys.path:
        sys.path.append(ROOT_DIR)
    import metrics

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # limit to 5MB

def analyze(csv, save_local=False):

    # store html result into a local html file

    if save_local:

        fn = os.path.dirname(os.path.realpath(__file__)) + \
            "/" + str(uuid.uuid4()) + ".html"

        with open(fn, 'w') as f:
            f.write(metrics.analyze_file(csv))

        # fn is the local save path

    return metrics.analyze_file(csv)  # return the html content


@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("home.html")


@app.route("/submit", methods=['GET', 'POST'])
def run_ackl():    
    if request.method == 'POST':

        r = ''
        use_sample = request.form["use_sample"]

        if (use_sample):
            # path = "ackl/gui/static/sample.csv"
            X, y, X_names, desc, labels = io.load_dataset('salt', x_range=list(range(400, 1000)), display=False)
            pkl = metrics.preview_kernels(X, y, metrics=True, 
                                          scatterplot=False, 
                                          selected_kernel_names=['linear'])
            r = metrics.visualize_metric_dicts(pkl, plot=False)
        else:
            pass

    return {'message': 'success', 'html': r}
    #render_template("home.html", use_sample = use_sample,  cla_result = r)

# def open_browser():
#    webbrowser.open_new('http://localhost:5052/')

if __name__ == '__main__':
    # # use netstat -ano|findstr 5051 to check port use
    # Timer(3, open_browser).start()
    # app.run(host="0.0.0.0", port=5052, debug=False)
    FlaskUI(app=app, server="flask", port=5052).run()