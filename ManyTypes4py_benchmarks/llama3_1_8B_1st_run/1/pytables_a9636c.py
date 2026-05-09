from typing import Any, Dict, List, Optional, Tuple, Union

class TableIterator:
    def __init__(
        self,
        store: HDFStore,
        s: Table,
        func: Callable,
        where: Union[None, List[Term], Tuple[Term, ...]],
        nrows: Optional[int],
        start: Optional[int],
        stop: Optional[int],
        iterator: bool,
        chunksize: Optional[int],
        auto_close: bool,
    ) -> None:
        self.store = store
        self.s = s
        self.func = func
        self.where = where
        self.nrows = nrows
        self.start = start
        self.stop = stop
        self.coordinates = None
        self.chunksize = chunksize
        self.auto_close = auto_close

class IndexCol:
    def __init__(
        self,
        name: str,
        values: Optional[np.ndarray],
        kind: Optional[str],
        typ: Optional[Any],
        cname: Optional[str],
        axis: Optional[int],
        pos: Optional[int],
        freq: Optional[str],
        tz: Optional[str],
        index_name: Optional[str],
        ordered: Optional[bool],
        table: Optional[Table],
        meta: Optional[str],
        metadata: Optional[np.ndarray],
    ) -> None:
        self.values = values
        self.kind = kind
        self.typ = typ
        self.name = name
        self.cname = cname or name
        self.axis = axis
        self.pos = pos
        self.freq = freq
        self.tz = tz
        self.index_name = index_name
        self.ordered = ordered
        self.table = table
        self.meta = meta
        self.metadata = metadata

class DataCol(IndexCol):
    def __init__(
        self,
        name: str,
        values: Optional[np.ndarray],
        kind: Optional[str],
        typ: Optional[Any],
        cname: Optional[str],
        pos: Optional[int],
        tz: Optional[str],
        ordered: Optional[bool],
        table: Optional[Table],
        meta: Optional[str],
        metadata: Optional[np.ndarray],
        dtype: Optional[str],
        data: Optional[np.ndarray],
    ) -> None:
        super().__init__(
            name=name,
            values=values,
            kind=kind,
            typ=typ,
            cname=cname,
            pos=pos,
            tz=tz,
            ordered=ordered,
            table=table,
            meta=meta,
            metadata=metadata,
        )
        self.dtype = dtype
        self.data = data

class GenericIndexCol(IndexCol):
    def __init__(
        self,
        name: str,
        values: Optional[np.ndarray],
        kind: Optional[str],
        typ: Optional[Any],
        cname: Optional[str],
        axis: Optional[int],
        pos: Optional[int],
        freq: Optional[str],
        tz: Optional[str],
        index_name: Optional[str],
        ordered: Optional[bool],
        table: Optional[Table],
        meta: Optional[str],
        metadata: Optional[np.ndarray],
    ) -> None:
        super().__init__(
            name=name,
            values=values,
            kind=kind,
            typ=typ,
            cname=cname,
            axis=axis,
            pos=pos,
            freq=freq,
            tz=tz,
            index_name=index_name,
            ordered=ordered,
            table=table,
            meta=meta,
            metadata=metadata,
        )

class DataIndexableCol(DataCol):
    def __init__(
        self,
        name: str,
        values: Optional[np.ndarray],
        kind: Optional[str],
        typ: Optional[Any],
        cname: Optional[str],
        pos: Optional[int],
        tz: Optional[str],
        ordered: Optional[bool],
        table: Optional[Table],
        meta: Optional[str],
        metadata: Optional[np.ndarray],
        dtype: Optional[str],
        data: Optional[np.ndarray],
    ) -> None:
        super().__init__(
            name=name,
            values=values,
            kind=kind,
            typ=typ,
            cname=cname,
            pos=pos,
            tz=tz,
            ordered=ordered,
            table=table,
            meta=meta,
            metadata=metadata,
            dtype=dtype,
            data=data,
        )

class GenericDataIndexableCol(DataIndexableCol):
    def __init__(
        self,
        name: str,
        values: Optional[np.ndarray],
        kind: Optional[str],
        typ: Optional[Any],
        cname: Optional[str],
        pos: Optional[int],
        tz: Optional[str],
        ordered: Optional[bool],
        table: Optional[Table],
        meta: Optional[str],
        metadata: Optional[np.ndarray],
        dtype: Optional[str],
        data: Optional[np.ndarray],
    ) -> None:
        super().__init__(
            name=name,
            values=values,
            kind=kind,
            typ=typ,
            cname=cname,
            pos=pos,
            tz=tz,
            ordered=ordered,
            table=table,
            meta=meta,
            metadata=metadata,
            dtype=dtype,
            data=data,
        )

class GenericFixed(Fixed):
    def __init__(
        self,
        parent: HDFStore,
        group: tables.Node,
        encoding: Optional[str],
        errors: Optional[str],
    ) -> None:
        super().__init__(parent, group, encoding=encoding, errors=errors)

class SeriesFixed(GenericFixed):
    def __init__(
        self,
        parent: HDFStore,
        group: tables.Node,
        encoding: Optional[str],
        errors: Optional[str],
    ) -> None:
        super().__init__(parent, group, encoding=encoding, errors=errors)

class BlockManagerFixed(GenericFixed):
    def __init__(
        self,
        parent: HDFStore,
        group: tables.Node,
        encoding: Optional[str],
        errors: Optional[str],
    ) -> None:
        super().__init__(parent, group, encoding=encoding, errors=errors)

class FrameFixed(BlockManagerFixed):
    def __init__(
        self,
        parent: HDFStore,
        group: tables.Node,
        encoding: Optional[str],
        errors: Optional[str],
    ) -> None:
        super().__init__(parent, group, encoding=encoding, errors=errors)

class Table(Fixed):
    def __init__(
        self,
        parent: HDFStore,
        group: tables.Node,
        encoding: Optional[str],
        errors: Optional[str],
        index_axes: Optional[List[IndexCol]],
        non_index_axes: Optional[List[Tuple[int, List[str]]]],
        values_axes: Optional[List[DataCol]],
        data_columns: Optional[List[str]],
        info: Optional[Dict[str, Dict[str, Any]]],
        nan_rep: Optional[str],
    ) -> None:
        super().__init__(parent, group, encoding=encoding, errors=errors)
        self.index_axes = index_axes
        self.non_index_axes = non_index_axes
        self.values_axes = values_axes
        self.data_columns = data_columns
        self.info = info
        self.nan_rep = nan_rep

class WORMTable(Table):
    def __init__(
        self,
        parent: HDFStore,
        group: tables.Node,
        encoding: Optional[str],
        errors: Optional[str],
        index_axes: Optional[List[IndexCol]],
        non_index_axes: Optional[List[Tuple[int, List[str]]]],
        values_axes: Optional[List[DataCol]],
        data_columns: Optional[List[str]],
        info: Optional[Dict[str, Dict[str, Any]]],
        nan_rep: Optional[str],
    ) -> None:
        super().__init__(
            parent,
            group,
            encoding=encoding,
            errors=errors,
            index_axes=index_axes,
            non_index_axes=non_index_axes,
            values_axes=values_axes,
            data_columns=data_columns,
            info=info,
            nan_rep=nan_rep,
        )

class AppendableTable(Table):
    def __init__(
        self,
        parent: HDFStore,
        group: tables.Node,
        encoding: Optional[str],
        errors: Optional[str],
        index_axes: Optional[List[IndexCol]],
        non_index_axes: Optional[List[Tuple[int, List[str]]]],
        values_axes: Optional[List[DataCol]],
        data_columns: Optional[List[str]],
        info: Optional[Dict[str, Dict[str, Any]]],
        nan_rep: Optional[str],
    ) -> None:
        super().__init__(
            parent,
            group,
            encoding=encoding,
            errors=errors,
            index_axes=index_axes,
            non_index_axes=non_index_axes,
            values_axes=values_axes,
            data_columns=data_columns,
            info=info,
            nan_rep=nan_rep,
        )

class AppendableFrameTable(AppendableTable):
    def __init__(
        self,
        parent: HDFStore,
        group: tables.Node,
        encoding: Optional[str],
        errors: Optional[str],
        index_axes: Optional[List[IndexCol]],
        non_index_axes: Optional[List[Tuple[int, List[str]]]],
        values_axes: Optional[List[DataCol]],
        data_columns: Optional[List[str]],
        info: Optional[Dict[str, Dict[str, Any]]],
        nan_rep: Optional[str],
    ) -> None:
        super().__init__(
            parent,
            group,
            encoding=encoding,
            errors=errors,
            index_axes=index_axes,
            non_index_axes=non_index_axes,
            values_axes=values_axes,
            data_columns=data_columns,
            info=info,
            nan_rep=nan_rep,
        )

class AppendableSeriesTable(AppendableFrameTable):
    def __init__(
        self,
        parent: HDFStore,
        group: tables.Node,
        encoding: Optional[str],
        errors: Optional[str],
        index_axes: Optional[List[IndexCol]],
        non_index_axes: Optional[List[Tuple[int, List[str]]]],
        values_axes: Optional[List[DataCol]],
        data_columns: Optional[List[str]],
        info: Optional[Dict[str, Dict[str, Any]]],
        nan_rep: Optional[str],
    ) -> None:
        super().__init__(
            parent,
            group,
            encoding=encoding,
            errors=errors,
            index_axes=index_axes,
            non_index_axes=non_index_axes,
            values_axes=values_axes,
            data_columns=data_columns,
            info=info,
            nan_rep=nan_rep,
        )

class AppendableMultiSeriesTable(AppendableSeriesTable):
    def __init__(
        self,
        parent: HDFStore,
        group: tables.Node,
        encoding: Optional[str],
        errors: Optional[str],
        index_axes: Optional[List[IndexCol]],
        non_index_axes: Optional[List[Tuple[int, List[str]]]],
        values_axes: Optional[List[DataCol]],
        data_columns: Optional[List[str]],
        info: Optional[Dict[str, Dict[str, Any]]],
        nan_rep: Optional[str],
    ) -> None:
        super().__init__(
            parent,
            group,
            encoding=encoding,
            errors=errors,
            index_axes=index_axes,
            non_index_axes=non_index_axes,
            values_axes=values_axes,
            data_columns=data_columns,
            info=info,
            nan_rep=nan_rep,
        )

class AppendableMultiFrameTable(AppendableFrameTable):
    def __init__(
        self,
        parent: HDFStore,
        group: tables.Node,
        encoding: Optional[str],
        errors: Optional[str],
        index_axes: Optional[List[IndexCol]],
        non_index_axes: Optional[List[Tuple[int, List[str]]]],
        values_axes: Optional[List[DataCol]],
        data_columns: Optional[List[str]],
        info: Optional[Dict[str, Dict[str, Any]]],
        nan_rep: Optional[str],
    ) -> None:
        super().__init__(
            parent,
            group,
            encoding=encoding,
            errors=errors,
            index_axes=index_axes,
            non_index_axes=non_index_axes,
            values_axes=values_axes,
            data_columns=data_columns,
            info=info,
            nan_rep=nan_rep,
        )

class GenericTable(AppendableFrameTable):
    def __init__(
        self,
        parent: HDFStore,
        group: tables.Node,
        encoding: Optional[str],
        errors: Optional[str],
        index_axes: Optional[List[IndexCol]],
        non_index_axes: Optional[List[Tuple[int, List[str]]]],
        values_axes: Optional[List[DataCol]],
        data_columns: Optional[List[str]],
        info: Optional[Dict[str, Dict[str, Any]]],
        nan_rep: Optional[str],
    ) -> None:
        super().__init__(
            parent,
            group,
            encoding=encoding,
            errors=errors,
            index_axes=index_axes,
            non_index_axes=non_index_axes,
            values_axes=values_axes,
            data_columns=data_columns,
            info=info,
            nan_rep=nan_rep,
        )

class Selection:
    def __init__(self, table: Table, where: Union[None, List[Term], Tuple[Term, ...]], start: Optional[int], stop: Optional[int]) -> None:
        self.table = table
        self.where = where
        self.start = start
        self.stop = stop
        self.condition = None
        self.filter = None
        self.terms = None
        self.coordinates = None

    def generate(self, where: Union[None, List[Term], Tuple[Term, ...]]) -> Optional[PyTablesExpr]:
        if where is None:
            return None
        q = self.table.queryables()
        try:
            return PyTablesExpr(where, queryables=q, encoding=self.table.encoding)
        except NameError as err:
            qkeys = ','.join(q.keys())
            msg = dedent(
                f"The passed where expression: {where}\ncontains an invalid variable reference\nall of the variable references must be a reference to\nan axis (e.g. 'index' or 'columns'), or a data_column\nThe currently defined references are: {qkeys}\n"
            )
            raise ValueError(msg) from err

    def select(self) -> np.ndarray:
        if self.condition is not None:
            return self.table.table.read_where(self.condition.format(), start=self.start, stop=self.stop)
        elif self.coordinates is not None:
            return self.table.table.read_coordinates(self.coordinates)
        return self.table.table.read(start=self.start, stop=self.stop)

    def select_coords(self) -> np.ndarray:
        start, stop = (self.start, self.stop)
        nrows = self.table.nrows
        if start is None:
            start = 0
        elif start < 0:
            start += nrows
        if stop is None:
            stop = nrows
        elif stop < 0:
            stop += nrows
        if self.condition is not None:
            return self.table.table.get_where_list(self.condition.format(), start=start, stop=stop, sort=True)
        elif self.coordinates is not None:
            return self.coordinates
        return np.arange(start, stop)
