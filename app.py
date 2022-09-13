from distutils.log import debug
from xmlrpc.client import TRANSPORT_ERROR
from flask import Flask,render_template,request
from artifacts.utils import data_predict


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/boston',methods=['POST'])
def boston_data():
    input_data = request.form
    res = ""
    print(input_data)
    price_predict = data_predict(input_data)
    res_data = price_predict.predict()
    return render_template('index.html',price=res_data)


if __name__ == "__main__":
    app.run(port=8080,debug=True)