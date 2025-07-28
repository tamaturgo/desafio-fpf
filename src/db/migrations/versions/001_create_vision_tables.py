from alembic import op
import sqlalchemy as sa


revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'vision_results',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('task_id', sa.String(), unique=True, nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime()),
        sa.Column('result', sa.JSON()),
    )
    op.create_table(
        'vision_tasks',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('task_id', sa.String(), unique=True, nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime()),
        sa.Column('has_result', sa.String()),
    )

def downgrade():
    op.drop_table('vision_results')
    op.drop_table('vision_tasks')
