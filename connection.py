import pandas as pd
import sqlite3

def dbconn():
    # Path to input database
    SQL_PATH = 'data/ForestFireDB.db'

    # Path to model storage location
    MODEL_STORAGE_PATH = 'models/'

    # Read in training data
    conn = sqlite3.connect(SQL_PATH)
    read_query = '''SELECT  x,
                            y,
                            month,
                            day,
                            FFMC,
                            DMC,
                            DC,
                            ISI,
                            temp,
                            RH,
                            wind,
                            rain,
                            area
                     FROM fires'''
    db_data = pd.read_sql(read_query, conn)
    conn.close()

    return db_data
