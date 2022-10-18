import flask
from flask_cors import CORS
from flask import jsonify
from flask import request
from pathlib import Path
from ModelTrain.ModelPredictions import Predictions

import urllib.parse as up
import jieba
import jieba.analyse
import re

app = flask.Flask(__name__)
CORS(app)
path = Path(__file__)
par_path = path.parent.parent

prediction = Predictions()
@app.route('/', methods=['POST'])
def detect():
    text = request.get_json()
    sentence = up.unquote(text['text'])
    chose = up.unquote(text['model'])
    cate = prediction.run(sentence, chose)
    # cate = sub['Category'].item()
    value = []

    if cate == 'budget':
        pattern = re.compile('\d+')
        number = pattern.search(sentence)
        if number:
            value.append(number.group(0))
    elif cate == "fixed_cate" or cate == 'account_cate' or cate == 'income_cate':
        value = fetch_cate_and_money(sentence)
        value.insert(1, detect_subcate_category(value[0]))
        if cate == 'account_cate':
            value.insert(0, '支出')
        elif cate == 'income_cate':
            value.insert(0, '收入')
    elif cate == 'account':
        value.append('支出')
    elif cate == 'income':
        value.append('收入')

    elif cate == 'category':
        value = format_new_category(sentence)

    result = {
        'model': chose,
        'TEXT': sentence,
        'DetectResult': cate,
        'value': value,
    }

    return jsonify(result)


def detect_subcate_category(subcate):
    sub = prediction.run(subcate, 1)
    cate = sub['Category'].item()
    return cate


def format_new_category(sentence):
    cate = jieba.analyse.extract_tags(sentence, 2)
    if ("分類" in cate):
        cate.remove("分類")
    else:
        if len(cate) > 1:
            cate.remove(cate[1])
    return cate


def fetch_cate_and_money(sentence):
    result = []
    pattern = re.compile('\d+')
    number = pattern.search(sentence)
    cate = jieba.analyse.extract_tags(sentence, 1)
    result.append(cate[0])
    if number:
        result.append(number.group(0))

    return result


if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(host='0.0.0.0', port=5000)
