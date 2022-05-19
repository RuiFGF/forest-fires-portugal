import pandas as pd
import sqlite3

# Path to csv file
CSV_PATH = 'data/heart.csv'

# Path to sqlite databse
SQL_PATH = 'data/HeartFailureDB.db'

# Read csv
data = pd.read_csv(CSV_PATH)
print("shape of input data is " + str(data.shape) )
data.head()



# Code in the following cells was adapted from: https://www.sqlitetutorial.net/sqlite-python/create-tables/
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


create_table_str = """CREATE TABLE IF NOT EXISTS heart (
                                    id integer PRIMARY KEY,
                                    Age integer,
                                    Sex text,
                                    ChestPainType text,
                                    RestingBP integer,
                                    Cholesterol integer,
                                    FastingBS integer,
                                    RestingECG text,
                                    MaxHR integer,
                                    ExerciseAngina text,
                                    Oldpeak real,
                                    ST_Slope text,
                                    HeartDisease integer
                                );"""

# Create connection
conn = create_connection(SQL_PATH)

# Create table
if conn is not None:
    create_table(conn, create_table_str)

conn.close()



# Write pandas data to sql table
conn = create_connection(SQL_PATH)
data.to_sql('heart', conn, if_exists='append', index=False)
conn.close()



# Test read
conn = create_connection(SQL_PATH)
data_sql = pd.read_sql("SELECT * FROM heart", conn) # Never use SELECT * in production
conn.close()

print("shape of output data is " + str(data_sql.shape) )
data_sql.head()