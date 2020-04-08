from peewee import *

db = SqliteDatabase('orders_and_couriers.db')


class Courier(Model):
    courier_id = IntegerField()
    locationX = DoubleField()
    locationY = DoubleField()

    class Meta:
        database = db


class Order(Model):
    client_id = IntegerField()
    priority = CharField()  # TODO add enum for priority
    status = IntegerField()  # TODO add enum for status
    text = CharField()
    courier = IntegerField()
    locationX = DoubleField()
    locationY = DoubleField()
    order_id = IntegerField()

    class Meta:
        database = db


class Global(Model):
    id = IntegerField()
    root = CharField()

    class Meta:
        database = db
