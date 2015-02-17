from flask import Flask, render_template, request, url_for, request, redirect
from flask.ext.sqlalchemy import SQLAlchemy
import sqlite3
import os

app = Flask(__name__)
sqlite3.connect(os.path.abspath("test.db"))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///test.db"
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    email = db.Column(db.String(120), unique=True)

    def __init__(self, username, email):
        self.username = username
        self.email = email
    def __repr__(self):
        return "<User %r>" % self.username


db.create_all()
x = User("tes2t", "test2@gmail.com")
db.session.add(x)
db.session.commit()


