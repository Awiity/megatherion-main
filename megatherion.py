import copy
from abc import abstractmethod, ABC
from json import load
from numbers import Real
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple, Union, Any, List, Callable
from enum import Enum
from collections.abc import MutableSequence
import heapq


class Type(Enum):
    Float = 0
    String = 1


def to_float(obj) -> float:
    """
    casts object to float with support of None objects (None is cast to None)
    """
    return float(obj) if obj is not None else None


def to_str(obj) -> str:
    """
    casts object to float with support of None objects (None is cast to None)
    """
    return str(obj) if obj is not None else None


def common(iterable):  # from ChatGPT
    """
    returns True if all items of iterable are the same.
    :param iterable:
    :return:
    """
    try:
        # Nejprve zkusíme získat první prvek iterátoru
        iterator = iter(iterable)
        first_value = next(iterator)
    except StopIteration:
        # Vyvolá výjimku, pokud je iterátor prázdný
        raise ValueError("Iterable is empty")

    # Kontrola, zda jsou všechny další prvky stejné jako první prvek
    for value in iterator:
        if value != first_value:
            raise ValueError("Not all values are the same")

    # Vrací hodnotu, pokud všechny prvky jsou stejné
    return first_value


class Column(MutableSequence):  # implement MutableSequence (some method are mixed from abc)
    """
    Representation of column of dataframe. Column has datatype: float columns contains
    only floats and None values, string columns contains strings and None values.
    """

    def __init__(self, data: Iterable, dtype: Type):
        self.dtype = dtype
        self._cast = to_float if self.dtype == Type.Float else to_str
        # cast function (it casts to floats for Float datatype or
        # to strings for String datattype)
        self._data = [self._cast(value) for value in data]

    def __len__(self) -> int:
        """
        Implementation of abstract base class `MutableSequence`.
        :return: number of rows
        """
        return len(self._data)

    def __getitem__(self, item: Union[int, slice]) -> Union[float,
                                                            str, list[str], list[float]]:
        """
        Indexed getter (get value from index or sliced sublist for slice).
        Implementation of abstract base class `MutableSequence`.
        :param item: index or slice
        :return: item or list of items
        """
        return self._data[item]

    def __setitem__(self, key: Union[int, slice], value: Any) -> None:
        """
        Indexed setter (set value to index, or list to sliced column)
        Implementation of abstract base class `MutableSequence`.
        :param key: index or slice
        :param value: simple value or list of values

        """
        self._data[key] = self._cast(value)

    def append(self, item: Any) -> None:
        """
        Item is appended to column (value is cast to float or string if is not number).
        Implementation of abstract base class `MutableSequence`.
        :param item: appended value
        """
        self._data.append(self._cast(item))

    def insert(self, index: int, value: Any) -> None:
        """
        Item is inserted to colum at index `index` (value is cast to float or string if is not number).
        Implementation of abstract base class `MutableSequence`.
        :param index:  index of new item
        :param value:  inserted value
        :return:
        """
        self._data.insert(index, self._cast(value))

    def __delitem__(self, index: Union[int, slice]) -> None:
        """
        Remove item from index `index` or sublist defined by `slice`.
        :param index: index or slice
        """
        del self._data[index]

    def permute(self, indices: List[int]) -> 'Column':
        """
        Return new column which items are defined by list of indices (to original column).
        (eg. `Column(["a", "b", "c"]).permute([0,0,2])`
        returns  `Column(["a", "a", "c"])
        :param indices: list of indexes (ints between 0 and len(self) - 1)
        :return: new column
        """
        assert len(indices) == len(self)
        ...

    def copy(self) -> 'Column':
        """
        Return shallow copy of column.
        :return: new column with the same items
        """
        # FIXME: value is cast to the same type (minor optimisation problem)
        return Column(self._data, self.dtype)

    def get_formatted_item(self, index: int, *, width: int):
        """
        Auxiliary(вспомогательный) method for formating column items to string with `width`
        characters. Numbers (floats) are right aligned and strings left aligned.
        Nones are formatted as aligned "n/a".
        :param index: index of item
        :param width:  width
        :return:
        """
        assert width > 0
        if self._data[index] is None:
            if self.dtype == Type.Float:
                return "n/a".rjust(width)
            else:
                return "n/a".ljust(width)
        return format(self._data[index],
                      f"{width}s" if self.dtype == Type.String else f"-{width}.2g")


class DataFrame:
    """
    Dataframe with typed and named columns
    """

    def __init__(self, columns: Dict[str, Column]):
        """
        :param columns: columns of dataframe (key: name of dataframe),
                        lengths of all columns has to be the same
        """
        assert len(columns) > 0, "Dataframe without columns is not supported"
        self._size = common(len(column) for column in columns.values())
        # deep copy od dict `columns`
        self._columns = {name: column.copy() for name, column in columns.items()}

    def __getitem__(self, index: int) -> Tuple[Union[str, float]]:
        """
        Indexed getter returns row of dataframe as tuple
        :param index: index of row
        :return: tuple of items in row
        """
        try:
            return tuple(ele[index] for ele in self._columns.values())
        except IndexError:
            print(f"IndexError: row is out of index")

    def __iter__(self) -> Iterator[Tuple[Union[str, float]]]:
        """
        :return: iterator over rows of dataframe
        """
        for i in range(len(self)):
            yield tuple(c[i] for c in self._columns.values())   # yield keyword is used for more memory efficient code

    def __len__(self) -> int:
        """
        :return: count of rows
        """
        return self._size

    @property
    def columns(self) -> Iterable[str]:
        """
        :return: names of columns (as iterable object)
        """
        return self._columns.keys()

    def __repr__(self) -> str:
        """
        :return: string representation of dataframe (table with aligned columns)
        """
        lines = []
        lines.append(" ".join(f"{name:12s}" for name in self.columns))
        for i in range(len(self)):
            lines.append(" ".join(self._columns[cname].get_formatted_item(i, width=12)
                                  for cname in self.columns))
        return "\n".join(lines)

    def dot(self, other: 'DataFrame') -> 'DataFrame':
        ...

    def transpose(self) -> 'DataFrame': # FIXME doesnt work if number of rows is bigger than columns
        dataFrame_transposed = copy.deepcopy(self)
        count = iter(self._columns)
        for row in self:
            dataFrame_transposed._columns[next(count)] = Column(list(row), Type.Float)  # FIXME need to correctly assign Type.

        return dataFrame_transposed
        # for col in df.columns:
        #     print(df._columns[col][1])

    def product(self, axis: int = 0) -> 'DataFrame':
        if not axis:
            dataframe_product = DataFrame(dict(a=Column([0 for col in self._columns], Type.Float)))
            result: int = 1
            count = 0
            for col in self._columns:
                for i in range(len(self._columns[col])):
                    result *= self._columns[col][i]
                dataframe_product._columns["a"][count] = result
                count += 1
                result = 1
            return dataframe_product
        if axis:
            dataframe_product = DataFrame(dict(a=Column([0 for row in self], Type.Float)))
            result: int = 1
            count = 0
            for row in self:
                for i in range(len(row)):
                    result *= row[i]
                dataframe_product._columns["a"][count] = result
                count += 1
                result = 1
            return dataframe_product

    def to_replace(self, to_replace, value) -> None:
        if type(to_replace) != list:
            to_replace = [float(to_replace)]
        else:
            to_replace = [float(x) for x in to_replace]
        for col in self._columns:
            for ele in range(len(self._columns[col])):
                if self._columns[col][ele] == to_replace or self._columns[col][ele] in to_replace:
                    self._columns[col][ele] = value

    def diff(self, axis: int = 0):
        df_copy = copy.deepcopy(self)
        if not axis:
            for c in df_copy._columns:
                for i in range(1, len(df_copy._columns[c]), 1):
                    pass

        return df_copy

    def cumsum(self, axis: int = 0) -> 'DataFrame':
        df_copy = copy.deepcopy(self)
        cumsum: float = 0
        if not axis:
            for col in df_copy._columns:
                for i in range(0, len(df_copy._columns[col])):
                    if df_copy._columns[col][i] is not None:
                        cumsum += df_copy._columns[col][i]
                        df_copy._columns[col][i] = cumsum
                cumsum = 0
        else:
            for row in df_copy:
                for i in range(len(row)):
                    if row[i] is not None:
                        cumsum += row[i]
                        row[i] = cumsum
                cumsum = 0
        return df_copy


    def append_column(self, col_name: str, column: Column) -> None:
        """
        Appends new column to dataframe (its name has to be unique).
        :param col_name:  name of new column
        :param column: data of new column
        """
        if col_name in self._columns:
            raise ValueError("Duplicate column name")
        self._columns[col_name] = column.copy()

    def append_row(self, row: tuple) -> None:
        """
        Appends new row to dataframe.
        :param row: tuple of values for all columns
        """

        if len(row) != len(self.columns):
            print(f"Wrong size: length of dataframe - {len(self._columns)}, length of given row - {len(row)}")
            return 0
        count: int = 0
        for col in self._columns:
            #print(f"DB: {self._columns[col][0]} is {type(self._columns[col][0])} == {row[count]} is{type(row[count])}")
            if type(self._columns[col][0]) != type(row[count]):
                raise TypeError("WRONG TYPE OF DATA")
            count += 1
        i: int = 0

        for col, value in zip(self._columns, row):
            self._columns[col].append(value)
        self._size += 1

    def filter(self, col_name: str,
               predicate: Callable[[Union[float, str]], bool]) -> 'DataFrame':
        """
        Returns new dataframe with rows which values in column `col_name` returns
        True in function `predicate`.

        :param col_name: name of tested column
        :param predicate: testing function
        :return: new dataframe
        """
        filter_res = {name: [] for name in self._columns.keys()}
        indices_to_keep = []
        for i in range(len(self)):
            if predicate(self._columns[col_name][i]):
                indices_to_keep.append(i)
        #print(f"indices_to_keep: {indices_to_keep}")
        for name, col in self._columns.items():
            filter_res[name] = [col[i] for i in indices_to_keep]
        print(f"filter_res: {filter_res}")
        return DataFrame(filter_res)

    def sort(self, col_name: str, ascending=True) -> 'DataFrame':
        """
        Sort dataframe by column with `col_name` ascending or descending.
        :param col_name: name of key column
        :param ascending: direction of sorting
        :return: new dataframe
        """
        self_copy = copy.deepcopy(self)
        if self._columns[col_name].dtype is not Type.Float:
            raise TypeError("Cannot sort NON-Type.Float Columns")
        self_copy._columns[col_name] = Column(sorted(self_copy._columns[col_name][::]), Type.Float)
        return self_copy


    def describe(self) -> str:
        """
        similar to pandas but only with min, max and avg statistics for floats and count"
        :return: string with formatted decription
        """
        keys = [name for name in self._columns.keys() if self._columns[name].dtype is Type.Float]
        result = {
            "count": dict(),
            "min": dict(),
            "max": dict(),
            "avg": dict()
        }
        #   Does the calculation
        for col_name in keys:
            col_data = self._columns[col_name][::]
            result["count"][col_name] = len(col_data)
            result["min"][col_name] = min([ele for ele in col_data if ele is not None])
            result["max"][col_name] = max([ele for ele in col_data if ele is not None])
            result["avg"][col_name] = sum([ele for ele in col_data if ele is not None])/len(col_data)
        #   Creating and formatting final string
        result_str: str = ""
        i = 0
        for name in keys:
            result_str += f"Column description of {name}\n"
            for ele in result:

                result_str += f"{ele:>5}: {str(result[ele][name]):>8}\n"

        return result_str

    def inner_join(self, other: 'DataFrame', self_key_column: str,
                   other_key_column: str) -> 'DataFrame':
        """
            Inner join between self and other dataframe with join predicate
            `self.key_column == other.key_column`.

            Possible collision of column identifiers is resolved by prefixing `_other` to
            columns from `other` data table.
        """
        ...

    def setvalue(self, col_name: str, row_index: int, value: Any) -> None:
        """
        Set new value in dataframe.
        :param col_name:  name of culumns
        :param row_index: index of row
        :param value:  new value (value is cast to type of column)
        :return:
        """
        col = self._columns[col_name]
        col[row_index] = col._cast(value)

    @staticmethod
    def read_csv(path: Union[str, Path]) -> 'DataFrame':
        """
        Read dataframe by CSV reader
        """
        return CSVReader(path).read()

    @staticmethod
    def read_json(path: Union[str, Path]) -> 'DataFrame':
        """
        Read dataframe by JSON reader
        """
        return JSONReader(path).read()


class Reader(ABC):
    def __init__(self, path: Union[Path, str]):
        self.path = Path(path)

    @abstractmethod
    def read(self) -> DataFrame:
        raise NotImplemented("Abstract method")


class JSONReader(Reader):
    """
    Factory class for creation of dataframe by CSV file. CSV file must contain
    header line with names of columns.
    The type of columns should be inferred from types of their values (columns which
    contains only value has to be floats columns otherwise string columns),
    """

    def read(self) -> DataFrame:
        with open(self.path, "rt") as f:
            json = load(f)
        columns = {}
        for cname in json.keys():  # cyklus přes sloupce (= atributy JSON objektu)
            dtype = Type.Float if all(value is None or isinstance(value, Real)
                                      for value in json[cname]) else Type.String
            columns[cname] = Column(json[cname], dtype)
        return DataFrame(columns)


class CSVReader(Reader):
    """
    Factory class for creation of dataframe by JSON file. JSON file must contain
    one object with attributes which array values represents columns.
    The type of columns are inferred from types of their values (columns which
    contains only value is floats columns otherwise string columns),
    """

    def read(self) -> 'DataFrame':
        ...


# if __name__ == "__main__":
#     df = DataFrame(dict(
#         a=Column([None, 3.1415], Type.Float),
#         b=Column(["a", 2], Type.String),
#         c=Column(range(2), Type.Float)
#     ))
#     df.setvalue("a", 1, 42)
#     print(df)
#
#     df = DataFrame.read_json("data.json")
#     print(df)
#
# for line in df:
#     print(line)
#
# ###
