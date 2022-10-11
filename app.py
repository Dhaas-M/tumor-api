
from flask import Flask, render_template, request
import os
from modelFiles import *

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imgFile = request.files['img']
    path = 'static\images'+imgFile.filename
    imgFile.save(path)


    result = TestBT(path)
    ans = int(result["ans"])
    if ans == 1: result['ans'] = 'present'
    else: result['ans'] = 'absent'

    list1=[result]
    return render_template('output.html', list1=list1)


if __name__ == '__main__':
    app.run(port=3000, debug=True)

