from datetime import date, datetime
import re
import uuid
import pytest
import pandas as pd
from pandas.io.excel import ExcelWriter
from pathlib import Path
from typing import Optional, Dict, Any
from odf.table import Table

odf = pytest.importorskip('odf')

@pytest.fixture
def ext() -> str:
    return '.ods'

@pytest.fixture
def tmp_excel(ext: str, tmp_path: Path) -> str:
    tmp = tmp_path / f'{uuid.uuid4()}{ext}'
    tmp.touch()
    return str(tmp)

def test_write_append_mode_raises(tmp_excel: str) -> None:
    msg = 'Append mode is not supported with odf!'
    with pytest.raises(ValueError, match=msg):
        ExcelWriter(tmp_excel, engine='odf', mode='a')

@pytest.mark.parametrize('engine_kwargs', [None, {'kwarg': 1}])
def test_engine_kwargs(tmp_excel: str, engine_kwargs: Optional[Dict[str, Any]]) -> None:
    if engine_kwargs is not None:
        error = re.escape("OpenDocumentSpreadsheet() got an unexpected keyword argument 'kwarg'")
        with pytest.raises(TypeError, match=error):
            ExcelWriter(tmp_excel, engine='odf', engine_kwargs=engine_kwargs)
    else:
        with ExcelWriter(tmp_excel, engine='odf', engine_kwargs=engine_kwargs) as _:
            pass

def test_book_and_sheets_consistent(tmp_excel: str) -> None:
    with ExcelWriter(tmp_excel) as writer:
        assert writer.sheets == {}
        table = Table(name='test_name')
        writer.book.spreadsheet.addElement(table)
        assert writer.sheets == {'test_name': table}

@pytest.mark.parametrize(
    ['value', 'cell_value_type', 'cell_value_attribute', 'cell_value'],
    argvalues=[
        (True, 'boolean', 'boolean-value', 'true'),
        ('test string', 'string', 'string-value', 'test string'),
        (1, 'float', 'value', '1'),
        (1.5, 'float', 'value', '1.5'),
        (datetime(2010, 10, 10, 10, 10, 10), 'date', 'date-value', '2010-10-10T10:10:10'),
        (date(2010, 10, 10), 'date', 'date-value', '2010-10-10')
    ]
)
def test_cell_value_type(
    tmp_excel: str,
    value: Any,
    cell_value_type: str,
    cell_value_attribute: str,
    cell_value: str
) -> None:
    from odf.namespaces import OFFICENS
    from odf.table import TableCell, TableRow
    table_cell_name = TableCell().qname
    pd.DataFrame([[value]]).to_excel(tmp_excel, header=False, index=False)
    with pd.ExcelFile(tmp_excel) as wb:
        sheet = wb._reader.get_sheet_by_index(0)
        sheet_rows = sheet.getElementsByType(TableRow)
        sheet_cells = [x for x in sheet_rows[0].childNodes if hasattr(x, 'qname') and x.qname == table_cell_name]
        cell = sheet_cells[0]
        assert cell.attributes.get((OFFICENS, 'value-type')) == cell_value_type
        assert cell.attributes.get((OFFICENS, cell_value_attribute)) == cell_value
