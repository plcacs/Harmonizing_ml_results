from dbt.tests.util import AnyInteger

def no_stats() -> dict[typing.Text, dict[typing.Text, typing.Union[typing.Text,bool]]]:
    return {'has_stats': {'id': 'has_stats', 'label': 'Has Stats?', 'value': False, 'description': 'Indicates whether there are statistics for this table', 'include': False}}

def base_expected_catalog(project: Union[str, bool, None], role: Union[str, None, set[int]], id_type: Union[int, None, set, str], text_type: Union[int, None, set, str], time_type: Union[int, None, set, str], view_type: Union[str, None, set[int]], table_type: Union[str, None, set[int]], model_stats: Union[str, list[str], int], seed_stats: Union[None, str, list[typing.Any], list["SoftwareApplication"]]=None, case: Union[None, str, dict[str, typing.Any]]=None, case_columns: bool=False) -> dict[typing.Text, dict[typing.Text, dict[typing.Text, typing.Union[typing.Text,dict[typing.Text, typing.Union[str,None,set[int]]],dict[str, int],dict[int, typing.Union[int,None]],list[str],int,dict[, dict[typing.Text, typing.Union[AnyInteger,int,None,set,str]]]]]]]:
    if case is None:

        def case(x: Any):
            return x
    col_case = case if case_columns else lambda x: x
    if seed_stats is None:
        seed_stats = model_stats
    model_database = project.database
    my_schema_name = case(project.test_schema)
    alternate_schema = case(project.test_schema + '_test')
    expected_cols = {col_case('id'): {'name': col_case('id'), 'index': AnyInteger(), 'type': id_type, 'comment': None}, col_case('first_name'): {'name': col_case('first_name'), 'index': AnyInteger(), 'type': text_type, 'comment': None}, col_case('email'): {'name': col_case('email'), 'index': AnyInteger(), 'type': text_type, 'comment': None}, col_case('ip_address'): {'name': col_case('ip_address'), 'index': AnyInteger(), 'type': text_type, 'comment': None}, col_case('updated_at'): {'name': col_case('updated_at'), 'index': AnyInteger(), 'type': time_type, 'comment': None}}
    return {'nodes': {'model.test.model': {'unique_id': 'model.test.model', 'metadata': {'schema': my_schema_name, 'database': model_database, 'name': case('model'), 'type': view_type, 'comment': None, 'owner': role}, 'stats': model_stats, 'columns': expected_cols}, 'model.test.second_model': {'unique_id': 'model.test.second_model', 'metadata': {'schema': alternate_schema, 'database': project.database, 'name': case('second_model'), 'type': view_type, 'comment': None, 'owner': role}, 'stats': model_stats, 'columns': expected_cols}, 'seed.test.seed': {'unique_id': 'seed.test.seed', 'metadata': {'schema': my_schema_name, 'database': project.database, 'name': case('seed'), 'type': table_type, 'comment': None, 'owner': role}, 'stats': seed_stats, 'columns': expected_cols}}, 'sources': {'source.test.my_source.my_table': {'unique_id': 'source.test.my_source.my_table', 'metadata': {'schema': my_schema_name, 'database': project.database, 'name': case('seed'), 'type': table_type, 'comment': None, 'owner': role}, 'stats': seed_stats, 'columns': expected_cols}}}

def expected_references_catalog(project: Union[str, bool, None], role: Union[str, bool, None], id_type: Union[int, None, str, list[str]], text_type: Union[str, None, bool, typing.Any], time_type: Union[int, None, str, list[str]], view_type: Union[str, bool, None], table_type: Union[str, bool, None], model_stats: Union[str, dict[str, typing.Any], bool], bigint_type: Union[None, str]=None, seed_stats: Union[None, str, list[str]]=None, case: Union[None, str, util.image.models.ImageType]=None, case_columns: bool=False, view_summary_stats: Union[None, str]=None) -> dict[typing.Text, dict[typing.Text, dict[typing.Text, typing.Union[typing.Text,dict[typing.Text, typing.Union[str,bool,None]],dict[int, typing.Union[int,None]],int,None,dict[typing.Text, dict[typing.Text, typing.Union[int,None,str,list[str]]]]]]]]:
    if case is None:

        def case(x: Any):
            return x
    col_case = case if case_columns else lambda x: x
    if seed_stats is None:
        seed_stats = model_stats
    if view_summary_stats is None:
        view_summary_stats = model_stats
    model_database = project.database
    my_schema_name = case(project.test_schema)
    summary_columns = {'first_name': {'name': 'first_name', 'index': 1, 'type': text_type, 'comment': None}, 'ct': {'name': 'ct', 'index': 2, 'type': bigint_type, 'comment': None}}
    seed_columns = {'id': {'name': col_case('id'), 'index': 1, 'type': id_type, 'comment': None}, 'first_name': {'name': col_case('first_name'), 'index': 2, 'type': text_type, 'comment': None}, 'email': {'name': col_case('email'), 'index': 3, 'type': text_type, 'comment': None}, 'ip_address': {'name': col_case('ip_address'), 'index': 4, 'type': text_type, 'comment': None}, 'updated_at': {'name': col_case('updated_at'), 'index': 5, 'type': time_type, 'comment': None}}
    return {'nodes': {'seed.test.seed': {'unique_id': 'seed.test.seed', 'metadata': {'schema': my_schema_name, 'database': project.database, 'name': case('seed'), 'type': table_type, 'comment': None, 'owner': role}, 'stats': seed_stats, 'columns': seed_columns}, 'model.test.ephemeral_summary': {'unique_id': 'model.test.ephemeral_summary', 'metadata': {'schema': my_schema_name, 'database': model_database, 'name': case('ephemeral_summary'), 'type': table_type, 'comment': None, 'owner': role}, 'stats': model_stats, 'columns': summary_columns}, 'model.test.view_summary': {'unique_id': 'model.test.view_summary', 'metadata': {'schema': my_schema_name, 'database': model_database, 'name': case('view_summary'), 'type': view_type, 'comment': None, 'owner': role}, 'stats': view_summary_stats, 'columns': summary_columns}}, 'sources': {'source.test.my_source.my_table': {'unique_id': 'source.test.my_source.my_table', 'metadata': {'schema': my_schema_name, 'database': project.database, 'name': case('seed'), 'type': table_type, 'comment': None, 'owner': role}, 'stats': seed_stats, 'columns': seed_columns}}}