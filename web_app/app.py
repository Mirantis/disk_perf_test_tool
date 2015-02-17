from config import DATABASE_URI
from flask import Flask
from flask.ext.bootstrap import Bootstrap
from flask.ext.sqlalchemy import SQLAlchemy

app = Flask(__name__)
db = SQLAlchemy(app)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URI
Bootstrap(app)

