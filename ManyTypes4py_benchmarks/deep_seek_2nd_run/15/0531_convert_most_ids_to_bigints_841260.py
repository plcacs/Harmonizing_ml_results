from typing import Any, List
from django.db import migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.operations.base import Operation
from django.db.migrations.state import StateApps
from typing_extensions import override

class MigrateIdToBigint(Operation):
    """A helper to migrate the id of a given table to bigint.

    Necessary for many-to-many intermediate tables without models, due
    to https://code.djangoproject.com/ticket/32674"""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @override
    def state_forwards(self, app_label: str, state: StateApps) -> None:
        pass

    @override
    def database_forwards(self, app_label: str, schema_editor: BaseDatabaseSchemaEditor, from_state: StateApps, to_state: StateApps) -> None:
        model = from_state.apps.get_model(app_label, self.model_name)
        table_name = model._meta.db_table
        seq_name = table_name + '_id_seq'
        schema_editor.execute(f'ALTER TABLE {schema_editor.quote_name(table_name)} ALTER COLUMN id SET DATA TYPE bigint')
        schema_editor.execute(f'ALTER SEQUENCE {schema_editor.quote_name(seq_name)} AS bigint')

    @override
    def database_backwards(self, app_label: str, schema_editor: BaseDatabaseSchemaEditor, from_state: StateApps, to_state: StateApps) -> None:
        model = from_state.apps.get_model(app_label, self.model_name)
        table_name = model._meta.db_table
        seq_name = table_name + '_id_seq'
        schema_editor.execute(f'ALTER TABLE {schema_editor.quote_name(table_name)} ALTER COLUMN id SET DATA TYPE int')
        schema_editor.execute(f'ALTER SEQUENCE {schema_editor.quote_name(seq_name)} AS int')

    @override
    def describe(self) -> str:
        return f"Migrates the 'id' column of {self.model_name} and its sequence to be a bigint.  Requires that the table have no foreign keys."

class Migration(migrations.Migration):
    atomic: bool = False
    dependencies: List[tuple] = [('zerver', '0530_alter_useractivity_id_alter_useractivityinterval_id')]
    operations: List[Operation] = [
        MigrateIdToBigint('archivedattachment_messages'),
        MigrateIdToBigint('attachment_messages'),
        MigrateIdToBigint('attachment_scheduled_messages'),
        MigrateIdToBigint('defaultstreamgroup_streams'),
        MigrateIdToBigint('multiuseinvite_streams'),
        MigrateIdToBigint('preregistrationuser_streams'),
        MigrateIdToBigint('scheduledemail_users'),
        MigrateIdToBigint('userprofile_groups'),
        MigrateIdToBigint('userprofile_user_permissions'),
        migrations.AlterField(
            model_name='alertword',
            name='id',
            field=models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')
        ),
        migrations.AlterField(
            model_name='archivedattachment',
            name='id',
            field=models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')
        ),
        # ... (rest of the AlterField operations remain the same)
    ]
