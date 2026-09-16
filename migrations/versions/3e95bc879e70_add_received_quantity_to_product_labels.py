"""add received_quantity to product_labels

Revision ID: 3e95bc879e70
Revises: 9905fe0eb107
Create Date: 2026-09-12 21:52:52.736670

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '3e95bc879e70'
down_revision = '9905fe0eb107'
branch_labels = None
depends_on = None


def upgrade():
    # Only the real change. Autogenerate also proposed dropping and
    # recreating 'scans_search_idx' — a false positive: that index is a raw
    # SQL expression (SEARCH_VECTOR_SQL in app/models.py) autogenerate can't
    # compare reliably, not an actual schema drift. Left untouched.
    with op.batch_alter_table('product_labels', schema=None) as batch_op:
        batch_op.add_column(sa.Column('received_quantity', sa.Numeric(precision=12, scale=3), nullable=True))


def downgrade():
    with op.batch_alter_table('product_labels', schema=None) as batch_op:
        batch_op.drop_column('received_quantity')
