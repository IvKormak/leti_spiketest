# This Python file uses the following encoding: utf-8

# if __name__ == "__main__":
#     pass

import configparser as cfg
import sqlite3 as sql
from collections import namedtuple

TRACES_FILE = "traces.ini"

Trace = namedtuple('Trace', ["id", "trace_path", "trace_alias", "target_speed", "end_time", "commentary"])

class DBManager:
    def __init__(self):
        self.con = sql.connect("config.db")
        self.cur = self.con.cursor()

        query = "CREATE TABLE IF NOT EXISTS traces (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT UNIQUE, alias TEXT, speed INT, end_time INT, commentary TEXT)"
        self.cur.execute(query)

    def add_trace_entry(self, trace_alias:str, trace_path:str, target_speed:int, end_time:int, commentary:str = ""):

        query = f"INSERT INTO traces(path, alias, speed, end_time, commentary) VALUES ('{trace_path}', '{trace_alias}', {target_speed}, {end_time}, '{commentary}')"
        self.cur.execute(query)
        self.con.commit()

    def delete_trace_entry(self, path):
        query = f"DELETE FROM traces WHERE path = '{path}'"
        self.cur.execute(query)
        self.con.commit()

    def read_trace_entries(self):
        query = "SELECT * FROM traces"
        self.cur.execute(query)
        return [Trace(*res) for res in self.cur.fetchall()]

    def get_trace_by_path(self, path):
        query = f"SELECT * FROM traces WHERE path = '{path}'"
        self.cur.execute(query)
        return Trace(*self.cur.fetchall()[0])

    def update(self, commit):
        query = f"""UPDATE traces SET end_time = {commit.end_time}, commentary = '{commit.commentary}' WHERE id = '{commit.id}'"""
        self.cur.execute(query)
        self.con.commit()



if __name__ == "__main__":
    db = DBManager()
    #db.add_trace_entry(trace_alias="a", trace_path="b/b", target_speed=1000, end_time=30000, commentary="")
    trace = db.get_trace_by_path(path = "b/b")
    db.update(Trace(trace.id, trace.trace_path, trace.trace_alias, trace.target_speed, 321, ""))
    print(db.read_trace_entries())
    db.update(Trace(trace.id, trace.trace_path, trace.trace_alias, trace.target_speed, 123, ""))
    print(db.read_trace_entries())
