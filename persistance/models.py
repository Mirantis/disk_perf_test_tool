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
    param_4 = db.Column(db.Text())
    param_5 = db.Column(db.Text())
    param_6 = db.Column(db.Text())
    param_7 = db.Column(db.Text())
    param_8 = db.Column(db.Text())
    param_9 = db.Column(db.Text())
    param_10 = db.Column(db.Text())
    param_11 = db.Column(db.Text())
    param_12 = db.Column(db.Text())
    param_13 = db.Column(db.Text())
    param_14 = db.Column(db.Text())
    param_15 = db.Column(db.Text())
    param_16 = db.Column(db.Text())
    param_17 = db.Column(db.Text())
    param_18 = db.Column(db.Text())
    param_19 = db.Column(db.Text())
    param_20 = db.Column(db.Text())

    def __repr__(self):
        return self.param_1 + " " + self.param_2 + " " + self.param_3


class Lab(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True)
    url = db.Column(db.String(256))
    type = db.Column(db.String(4096))
    fuel_version = db.Column(db.String(64))
    ceph_version = db.Column(db.String(64))
    lab_general_info = db.Column(db.Text)
    lab_meta = db.Column(db.Text)


class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    build_id = db.Column(db.Integer, ForeignKey('build.id'))
    lab_id = db.Column(db.Integer, ForeignKey('lab.id'))
    date = db.Column(db.DateTime)
    param_combination_id = db.Column(db.Integer,
                                     ForeignKey('param_combination.id'))
    bandwith = db.Column(db.Float)
    meta = db.Column(db.String(4096))

    def __repr__(self):
        return str(self.bandwith) + " " + str(self.date)
