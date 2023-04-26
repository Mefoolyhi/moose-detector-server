import mysql.connector
from datetime import datetime
import os

try:
    mydb = mysql.connector.connect(
        host=os.environ['DB_host'],
        user=os.environ['DB_user'],
        password=os.environ['DB_password']
    )

    cur = mydb.cursor()
    cur.execute("USE DB")

    prediction_time = datetime.now()
    with open('moose.jpg', 'rb') as file:
        photo = file.read()
    sql_stmt = f"INSERT INTO predictions(prediction_time, photo) VALUES(%s,%s)"
    cur.execute(sql_stmt, (prediction_time.strftime("%Y-%m-%d %H:%M:%S"), photo))
    mydb.commit()
except mysql.connector.Error as error:
    print("Failed inserting BLOB data into MySQL table {}".format(error))

finally:
    if mydb.is_connected():
        cur.close()
        mydb.close()

