from sqlalchemy import ForeignKey
from web_app.app import db


class Build(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    build_id = db.Column(db.String(64))
    name = db.Column(db.String(64))
    md5 = db.Column(db.String(64))
    type = db.Column(db.Integer)

    def __repr__(self):
        return self.build_id + " " + self.name + " " + self.type


class Param(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64))
    type = db.Column(db.String(64))
    descr = db.Column(db.String(4096))


class ParamCombination(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    param_1 = db.Column(db.Text())
    param_2 = db.Column(db.Text())
    param_3 = db.Column(db.Text())

    def __repr__(self):
        return self.param_1 + " " + self.param_2 + " " + self.param_3


class Lab(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64))
    type = db.Column(db.String(4096))


class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    build_id = db.Column(db.Integer, ForeignKey('build.id'))
    lab_id = db.Column(db.Integer)
    date = db.Column(db.DateTime)
    param_combination_id = db.Column(db.Integer, ForeignKey('param_combination.id'))
    bandwith = db.Column(db.Float)
    meta = db.Column(db.String(4096))

    def __repr__(self):
        return str(self.bandwith) + " " + str(self.date)
