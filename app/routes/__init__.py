from flask import Flask

from app.routes.labels import labels_bp
from app.routes.orders import orders_bp
from app.routes.pages import pages_bp
from app.routes.products import products_bp
from app.routes.scans import scans_bp
from app.routes.v2 import v2_bp


def register_blueprints(app: Flask) -> None:
    app.register_blueprint(pages_bp)
    app.register_blueprint(scans_bp)
    app.register_blueprint(orders_bp)
    app.register_blueprint(labels_bp)
    app.register_blueprint(products_bp)
    app.register_blueprint(v2_bp)
