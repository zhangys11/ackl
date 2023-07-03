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
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # limit to 50MB, to avoid 413 error.

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("home.html")


@app.route("/submit", methods=['POST'])
def analyze():
    r = ''
    kernel_type = request.form['kernel_type']
    use_sample = request.form["use_sample"]

    if kernel_type == 'all':
        kernel_type = list(metrics.kernel_dict.keys())
    else:
        kernel_type = [kernel_type]

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
        desc = fn

    try:
        r += '<h6>Load dataset</h6>'
        if len(X.shape) == 2:

            if y is None:
                r += io.draw_average(X, X_names, output_html=True)
            else:
                r += io.draw_class_average(X, y, X_names, labels, output_html=True)

            r += io.scatter_plot(X, y, labels=labels, output_html=True)

        r += '<p>' + desc + '</p><hr/>'
        r += '<h6>Kernel Transformation</h6>'

        _, dic_test_accs, dic, s = metrics.classify_with_kernels(X, y, scale = True,
                                        do_cla=True,
                                        run_clfs=True,
                                        plots=True,
                                        logplot=True,
                                        output_html= True,
                                        selected_kernel_names = kernel_type)

        r += s
        r += metrics.visualize_metric_dicts(dic, plot=False)

    except Exception as e:
        r += '<b>' + str(e) + '</b>'

    return {'message': 'success', 'html': r}

if __name__ == '__main__':
    FlaskUI(app=app, server="flask", port=5052).run()
