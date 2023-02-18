import mysql.connector

try:
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password"
    )

    cur = mydb.cursor()
    cur.execute("USE DB")
    sql_stmt = f"SELECT * FROM predictions"
    cur.execute(sql_stmt)
    response = cur.fetchall()

    for row in response:
        print(row[0], row[1])

except mysql.connector.Error as error:
    print("Failed inserting BLOB data into MySQL table {}".format(error))

finally:
    if mydb.is_connected():
        cur.close()
        mydb.close()
