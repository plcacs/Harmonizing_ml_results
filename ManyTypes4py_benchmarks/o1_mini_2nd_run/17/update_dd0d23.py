from __future__ import annotations
import logging
from functools import partial
from typing import Any, Dict, List, Optional

from flask_appbuilder.models.sqla import Model
from superset import is_feature_enabled, security_manager
from superset.commands.base import BaseCommand
from superset.commands.database.exceptions import (
    DatabaseConnectionFailedError,
    DatabaseExistsValidationError,
    DatabaseInvalidError,
    DatabaseNotFoundError,
    DatabaseUpdateFailedError,
)
from superset.commands.database.ssh_tunnel.create import CreateSSHTunnelCommand
from superset.commands.database.ssh_tunnel.delete import DeleteSSHTunnelCommand
from superset.commands.database.ssh_tunnel.exceptions import SSHTunnelingNotEnabledError
from superset.commands.database.ssh_tunnel.update import UpdateSSHTunnelCommand
from superset.daos.database import DatabaseDAO
from superset.daos.dataset import DatasetDAO
from superset.databases.ssh_tunnel.models import SSHTunnel
from superset.db_engine_specs.base import GenericDBException
from superset.exceptions import OAuth2RedirectError
from superset.models.core import Database
from superset.utils import json
from superset.utils.decorators import on_error, transaction

logger = logging.getLogger(__name__)


class UpdateDatabaseCommand(BaseCommand):

    def __init__(self, model_id: int, data: Dict[str, Any]) -> None:
        self._properties: Dict[str, Any] = data.copy()
        self._model_id: int = model_id
        self._model: Optional[Database] = None

    @transaction(on_error=partial(on_error, reraise=DatabaseUpdateFailedError))
    def run(self) -> Database:
        self._model = DatabaseDAO.find_by_id(self._model_id)
        if not self._model:
            raise DatabaseNotFoundError()
        self.validate()
        if 'masked_encrypted_extra' in self._properties:
            self._properties['encrypted_extra'] = self._model.db_engine_spec.unmask_encrypted_extra(
                self._model.encrypted_extra, self._properties.pop('masked_encrypted_extra')
            )
            self._handle_oauth2()
        original_database_name: str = self._model.database_name
        database: Database = DatabaseDAO.update(self._model, self._properties)
        database.set_sqlalchemy_uri(database.sqlalchemy_uri)
        ssh_tunnel: Optional[SSHTunnel] = self._handle_ssh_tunnel(database)
        try:
            self._refresh_catalogs(database, original_database_name, ssh_tunnel)
        except OAuth2RedirectError:
            pass
        return database

    def _handle_oauth2(self) -> None:
        """
        Handle changes in OAuth2.
        """
        if not self._model:
            return
        if self._properties['encrypted_extra'] is None:
            self._model.purge_oauth2_tokens()
            return
        current_config: Dict[str, Any] = self._model.get_oauth2_config()
        if not current_config:
            return
        encrypted_extra: Dict[str, Any] = json.loads(self._properties['encrypted_extra'])
        new_config: Dict[str, Any] = encrypted_extra.get('oauth2_client_info', {})
        keys: set = {'id', 'scope', 'authorization_request_uri', 'token_request_uri'}
        for key in keys:
            if current_config.get(key) != new_config.get(key):
                self._model.purge_oauth2_tokens()
                break

    def _handle_ssh_tunnel(self, database: Database) -> Optional[SSHTunnel]:
        """
        Delete, create, or update an SSH tunnel.
        """
        if 'ssh_tunnel' not in self._properties:
            return None
        if not is_feature_enabled('SSH_TUNNELING'):
            raise SSHTunnelingNotEnabledError()
        current_ssh_tunnel: Optional[SSHTunnel] = DatabaseDAO.get_ssh_tunnel(database.id)
        ssh_tunnel_properties: Optional[Dict[str, Any]] = self._properties['ssh_tunnel']
        if ssh_tunnel_properties is None:
            if current_ssh_tunnel:
                DeleteSSHTunnelCommand(current_ssh_tunnel.id).run()
            return None
        if current_ssh_tunnel is None:
            return CreateSSHTunnelCommand(database, ssh_tunnel_properties).run()
        return UpdateSSHTunnelCommand(current_ssh_tunnel.id, ssh_tunnel_properties).run()

    def _get_catalog_names(self, database: Database, ssh_tunnel: Optional[SSHTunnel]) -> List[str]:
        """
        Helper method to load catalogs.
        """
        try:
            return database.get_all_catalog_names(force=True, ssh_tunnel=ssh_tunnel)
        except OAuth2RedirectError:
            raise
        except GenericDBException as ex:
            raise DatabaseConnectionFailedError() from ex

    def _get_schema_names(self, database: Database, catalog: Optional[str], ssh_tunnel: Optional[SSHTunnel]) -> List[str]:
        """
        Helper method to load schemas.
        """
        try:
            return database.get_all_schema_names(force=True, catalog=catalog, ssh_tunnel=ssh_tunnel)
        except OAuth2RedirectError:
            raise
        except GenericDBException as ex:
            raise DatabaseConnectionFailedError() from ex

    def _refresh_catalogs(
        self, database: Database, original_database_name: str, ssh_tunnel: Optional[SSHTunnel]
    ) -> None:
        """
        Add permissions for any new catalogs and schemas.
        """
        catalogs: List[Optional[str]] = (
            self._get_catalog_names(database, ssh_tunnel)
            if database.db_engine_spec.supports_catalog
            else [None]
        )
        for catalog in catalogs:
            try:
                schemas: List[str] = self._get_schema_names(database, catalog, ssh_tunnel)
                if catalog:
                    perm: str = security_manager.get_catalog_perm(original_database_name, catalog)
                    existing_pvm: Optional[Any] = security_manager.find_permission_view_menu('catalog_access', perm)
                    if not existing_pvm:
                        security_manager.add_permission_view_menu(
                            'catalog_access', security_manager.get_catalog_perm(database.database_name, catalog)
                        )
                        for schema in schemas:
                            security_manager.add_permission_view_menu(
                                'schema_access', security_manager.get_schema_perm(database.database_name, catalog, schema)
                            )
                        continue
            except DatabaseConnectionFailedError:
                if catalog:
                    logger.warning('Error processing catalog %s', catalog)
                    continue
                raise
            self._refresh_schemas(database, original_database_name, catalog, schemas)
            if original_database_name != database.database_name:
                self._rename_database_in_permissions(database, original_database_name, catalog, schemas)

    def _refresh_schemas(
        self, database: Database, original_database_name: str, catalog: Optional[str], schemas: List[str]
    ) -> None:
        """
        Add new schemas that don't have permissions yet.
        """
        for schema in schemas:
            perm: str = security_manager.get_schema_perm(original_database_name, catalog, schema)
            existing_pvm: Optional[Any] = security_manager.find_permission_view_menu('schema_access', perm)
            if not existing_pvm:
                new_name: str = security_manager.get_schema_perm(database.database_name, catalog, schema)
                security_manager.add_permission_view_menu('schema_access', new_name)

    def _rename_database_in_permissions(
        self, database: Database, original_database_name: str, catalog: Optional[str], schemas: List[str]
    ) -> None:
        new_catalog_perm_name: str = security_manager.get_catalog_perm(database.database_name, catalog)
        if catalog:
            perm: str = security_manager.get_catalog_perm(original_database_name, catalog)
            existing_pvm: Optional[Any] = security_manager.find_permission_view_menu('catalog_access', perm)
            if existing_pvm:
                existing_pvm.view_menu.name = new_catalog_perm_name
        for schema in schemas:
            new_schema_perm_name: str = security_manager.get_schema_perm(database.database_name, catalog, schema)
            perm: str = security_manager.get_schema_perm(original_database_name, catalog, schema)
            existing_pvm: Optional[Any] = security_manager.find_permission_view_menu('schema_access', perm)
            if existing_pvm:
                existing_pvm.view_menu.name = new_schema_perm_name
            for dataset in DatabaseDAO.get_datasets(database.id, catalog=catalog, schema=schema):
                dataset.catalog_perm = new_catalog_perm_name
                dataset.schema_perm = new_schema_perm_name
                for chart in DatasetDAO.get_related_objects(dataset.id)['charts']:
                    chart.catalog_perm = new_catalog_perm_name
                    chart.schema_perm = new_schema_perm_name

    def validate(self) -> None:
        database_name: Optional[str] = self._properties.get('database_name')
        if database_name:
            if not DatabaseDAO.validate_update_uniqueness(self._model_id, database_name):
                raise DatabaseInvalidError(exceptions=[DatabaseExistsValidationError()])
