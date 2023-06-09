#!/usr/bin/env python3.11
from datetime import datetime
import os
from flask import Flask, jsonify, request, make_response, Blueprint
from flask_sqlalchemy import SQLAlchemy
from flask_httpauth import HTTPBasicAuth
from sqlalchemy import exc
import codecs
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)
app.config.from_pyfile('config.py')
db = SQLAlchemy(app)
auth = HTTPBasicAuth()
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.yaml'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Moose Detector"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prediction_time = db.Column(db.DateTime, nullable=False)
    camera_id = db.Column(db.Integer, nullable=False)
    photo = db.Column(db.LargeBinary(length=(2 ** 32) - 1), nullable=False)
    lynx_count = db.Column(db.Integer, nullable=False, default=0)
    hog_count = db.Column(db.Integer, nullable=False, default=0)
    bear_count = db.Column(db.Integer, nullable=False, default=0)
    moose_count = db.Column(db.Integer, nullable=False, default=0)

    def as_dict(self):
        return {
            'id': self.id,
            'prediction_time': self.prediction_time,
            'camera_id': self.camera_id,
            'photo': codecs.encode(self.photo, encoding='base64').decode('utf-8'),
            'lynx_count': self.lynx_count,
            'hog_count': self.hog_count,
            'bear_count': self.bear_count,
            'moose_count': self.moose_count
        }

    @classmethod
    def jsonify_all(cls):
        return jsonify(predictions=[pred.as_dict() for pred in cls.query.all()])


def to_date(date_string):
    return datetime.strptime(date_string, "%Y-%m-%d").date()


@app.route('/results')
@auth.login_required
def get_response():
    date_to = request.args.get('dateTo', default=datetime.now().strftime("%Y-%m-%d"), type=to_date)
    date_from = request.args.get('dateFrom', default='2023-02-18', type=to_date)
    camera_id = request.args.get('cameraId', default=1, type=int)
    data = Prediction.query.filter_by(camera_id=camera_id).filter(
        Prediction.prediction_time.between(date_from, date_to)).all()
    return jsonify([pred.as_dict() for pred in data])


@app.route('/all')
@auth.login_required
def get_all_data():
    return Prediction.jsonify_all()


@app.route('/photo/<camera_id>', methods=['POST'])
@auth.login_required
def insert_picture(camera_id):
    try:
        prediction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('moose.jpg', 'rb') as file:
            photo = file.read()
        new_row = Prediction(camera_id=camera_id, prediction_time=prediction_time,
                             photo=photo, moose_count=1)
        db.session.add(new_row)
        db.session.commit()
        return make_response('', 204)
    except exc.SQLAlchemyError:
        return make_response('', 502)


@auth.verify_password
def authenticate(username, password):
    if username and password:
        if username == os.environ['USER'] and password == os.environ['PASSWORD']:
            return True
        else:
            return False
    return False


if __name__ == "__main__":
    app.run()
