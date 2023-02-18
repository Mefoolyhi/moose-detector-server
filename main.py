import mysql.connector
from datetime import datetime
from flask import Flask
import os
from flask import request, jsonify
from flask_httpauth import HTTPBasicAuth

app = Flask(__name__)
auth = HTTPBasicAuth()


async def get_results(date_from, date_to, camera_id):
    try:
        mydb = mysql.connector.connect(
            host=os.environ['DB_host'],
            user=os.environ['DB_user'],
            password=os.environ['DB_password']
        )

        cur = mydb.cursor()
        cur.execute("USE DB")
        sql_stmt = f"SELECT * FROM predictions WHERE camera_id = %s AND prediction_time BETWEEN %s AND %s"

        await cur.execute(sql_stmt, (camera_id, date_from, date_to))
        response = cur.fetchall()
        return response

    except mysql.connector.Error as error:
        return error
    finally:
        if mydb.is_connected():
            cur.close()
            mydb.close()


@app.route('/results')
@auth.login_required
async def get_response():
    date_to = request.args.get('dateTo', default=datetime.now().strftime("%Y-%m-%d"), type=str)
    date_from = request.args.get('dateFrom', default='2023-02-18', type=str)
    camera_id = request.args.get('cameraId', default=1, type=int)
    data = await get_results(date_from, date_to, camera_id)
    return jsonify(data)


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
