#!/usr/bin/env python3.9
from datetime import datetime
import os
from flask import Flask, jsonify, request, make_response, redirect, flash, Markup, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_httpauth import HTTPBasicAuth
from flask_admin import Admin, BaseView, expose
from flask_admin.contrib.sqla import ModelView
from sqlalchemy import exc
import codecs
from PIL import Image
from flask_swagger_ui import get_swaggerui_blueprint
from predictor import process_photo, process_video, stop_processing
from flask_babel import Babel
from log import log

app = Flask(__name__)
app.config.from_pyfile('config.py')
db = SQLAlchemy(app)
auth = HTTPBasicAuth()
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.yaml'
app.secret_key = b'a secret key'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Moose Detector"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)
app.config['FLASK_ADMIN_SWATCH'] = 'readable'
UPLOAD_FOLDER = 'photos_buffer'
ALLOWED_EXTENSIONS_IMAGE = {'png', 'jpg', 'jpeg'}
ALLOWED_EXTENSIONS_VIDEO = {'mp4', 'webm', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
babel = Babel(app)


def allowed_image_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_IMAGE


def allowed_image_video(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_VIDEO


class PredictionModelView(ModelView):
    page_size = 50
    can_create = False
    can_edit = False
    can_delete = False
    column_filters = ['camera_id', 'prediction_time', 'animal_type', 'animal_count']

    def _list_thumbnail(self, context, model, name):
        return Markup(
            '<img src="data:image/jpg;base64, ' +
            codecs.encode(model.photo, encoding='base64').decode('utf-8') + '" width="500"/>'
        )

    column_formatters = {
        'photo': _list_thumbnail
    }


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Вы не загрузили файл', 'error')
        file = request.files['file']
        if file:
            if allowed_image_file(file.filename):
                process_photo(Image.open(file), request.form['camera_id'], request.form['animal_type'])
                pred = stop_processing(request.form['camera_id'],
                                        request.form['prediction_date'])
                if insert_prediction(Prediction(**pred)):
                    flash('Файл загружен', 'success')
                else:
                    flash('Кокая-то ошипка, попробуйте потом, попейте чаюб', 'error')
            elif allowed_image_video(file.filename):
                pred = process_video(file.stream, request.form['camera_id'], request.form['animal_type'])
                if insert_prediction(Prediction(**pred)):
                    flash('Файл загружен', 'success')
                else:
                    flash('Кокая-то ошипка, попробуйте потом, попейте чаюб', 'error')
            else:
                flash('Недопустимый формат файла', 'error')
        else:
            flash('Файл пустой или не существует', 'error')
        return redirect('/upload')
    return redirect('admin/upload')


class UploadView(BaseView):
    @expose('/')
    def index(self):
        return self.render('upload_index.html')


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prediction_time = db.Column(db.DateTime, nullable=False)
    camera_id = db.Column(db.Integer, nullable=False)
    photo = db.Column(db.LargeBinary(length=(2 ** 32) - 1), nullable=False)
    animal_type = db.Column(db.String, nullable=False)
    animal_count = db.Column(db.Integer, nullable=False, default=0)

    def as_dict(self):
        return {
            'id': self.id,
            'prediction_time': self.prediction_time,
            'camera_id': self.camera_id,
            'photo': codecs.encode(self.photo, encoding='base64').decode('utf-8'),
            'animal_type': self.animal_type,
            'animal_count': self.animal_count
        }

    @classmethod
    def jsonify_all(cls):
        return jsonify(predictions=[pred.as_dict() for pred in cls.query.all()])


def to_date(date_string):
    return datetime.strptime(date_string, "%Y-%m-%d").date()


@app.route('/results')
#@auth.login_required
def get_response():
    date_to = request.args.get('dateTo', default=datetime.now().strftime("%Y-%m-%d"), type=to_date)
    date_from = request.args.get('dateFrom', default='2023-02-18', type=to_date)
    camera_id = request.args.get('cameraId', default=1, type=int)
    data = Prediction.query.filter_by(camera_id=camera_id).filter(
        Prediction.prediction_time.between(date_from, date_to)).all()
    return jsonify([pred.as_dict() for pred in data])


@app.route('/all')
#@auth.login_required
def get_all_data():
    return Prediction.jsonify_all()


@app.route('/photo/<camera_id>', methods=['GET', 'POST'])
#@auth.login_required
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


def insert_prediction(prediction):
    try:
        db.session.add(prediction)
        db.session.commit()
        return True
    except exc.SQLAlchemyError as e:
        print('Sorry Error while inserting')
        print(e)
        log('INSERT PREDICITION\n' + str(e))
        return False


@auth.verify_password
def authenticate(username, password):
    if username and password:
        if username == os.environ['USER'] and password == os.environ['PASSWORD']:
            return True
        else:
            return False
    return False


if __name__ == "__main__":
    admin = Admin(app, name='moose_detector', template_mode='bootstrap3')
    admin.add_view(PredictionModelView(Prediction, db.session))
    admin.add_view(UploadView(name='Upload', endpoint='upload'))
    app.run()
