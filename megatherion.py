import copy
import random
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
        # reseni
        result: list = []
        for index in indices:
            result.append(self._data[index])
        return Column(result, self.dtype)

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
        :param width:  widthw
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
        #print(f"asser {self._size} == {len(other._columns)} : {self._size == len(other._columns)}")
        assert self._size == len(other._columns)
        result_dict: dict = {}
        for other_name, other_col in other._columns.items():
            new_col_values: list = []

            for row_index in range(self._size):
                value_to_add = sum([other_col[i]*self[row_index][i] for i in range(len(self._columns))])
                new_col_values.append(value_to_add)
            result_dict.update({other_name: Column(new_col_values, Type.Float)})
        return DataFrame(result_dict)
        #WORKS HELLL YEAH


    def transpose(self) -> 'DataFrame': # Works only if all col.dtype 's are the same
        result_dict: dict = {}
        for i in range(self._size):
            new_col_values: list = []
            for col_name, col in self._columns.items():
                col_type = col.dtype
                new_col_values.append(col[i])
            new_col = Column(new_col_values, col_type)
            result_dict.update({str(i): new_col})
        return DataFrame(result_dict)

    def product(self, axis: int = 0) -> 'DataFrame':
        result_dict: dict = {}
        prod: float = 1
        result_col = Column([], Type.Float)
        if not axis:
            for name, col in self._columns.items():
                if col.dtype == Type.Float:
                    for ele in col[::]:
                        if ele:
                            prod *= ele
                    result_col.append(prod)
                else:
                    result_col.append(None)
                prod = 1.0
        else:
            for row in self:
                for ele in row:
                    print(type(ele))
                    if not (type(ele) is float):
                        """IF ANY OF THE COLUMNS IS Type.String THEN ALL PRODUCTS OF ROWS ARE n/a FOR EVERY ROW"""
                        [result_col.append(None) for i in self]
                        result_dict.update({"PRODUCT": result_col})
                        return DataFrame(result_dict)
                    else:
                        print("ASD")
                        prod *= ele
                result_col.append(prod)
                prod = 1.0
        result_dict.update({"PRODUCT": result_col})
        return DataFrame(result_dict)

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
        if axis == 0:
            for name, col in self._columns.items():

                for index in range(len(col) - 1, 0, -1):
                    col[index] = col[index] - col[index - 1]
                col[0] = None
        elif axis == 1:
            for index in range(self._size):
                temp_value = None
                for name, col in self._columns.items():
                    if not temp_value:
                        temp_value = col[index]
                        col[index] = None
                        continue
                    col[index] -= temp_value
                    temp_value = col[index] + temp_value


    def cumsum(self, axis: int = 0) -> 'DataFrame':
        result_dict: dict = {}
        if axis == 0:
            for name, col in self._columns.items():
                new_col_values: list = [col[0]]
                new_col_cumsum: int = col[0]
                for i in range(1, len(col[1:])+ 1):
                    new_col_cumsum += col[i]
                    new_col_values.append(new_col_cumsum)
                result_dict.update({name: Column(new_col_values, Type.Float)})
        elif axis == 1:
            rows = [{name: col[i] for name, col in self._columns.items()} for i in range(len(self))]
            for row in rows:
                row_cumsum: int = 0
                for name, value in row.items():
                    row_cumsum += value
                    row[name] = row_cumsum
            result_dict = {name: [row[name] for row in rows] for name in self.columns}
        for ele in result_dict:
            if type(result_dict[ele][0]) == float:
                result_dict[ele] = Column(result_dict[ele], Type.Float)
            else:
                result_dict[ele] = Column(result_dict[ele], Type.String)
        return DataFrame(result_dict)

    def unique(self, col_name: str) -> 'DataFrame':
        """A little too complicated, probably can do deepcopy of self, then
            del col[index] all shit and return copy
        """
        result_dict: dict = {}
        to_del: list = []
        for i in range(0,self._size,1):
            for j in range(i+1,self._size,1):
                if self._columns[col_name][j] == self._columns[col_name][i]:
                    if j not in to_del:
                        to_del.append(j)
        for name, col in self._columns.items():
            col_c = copy.deepcopy(col)
            for ele in to_del[::-1]:
                del col_c[ele]
            result_dict.update({name: col_c})

        return DataFrame(result_dict)

    def compare(self, other: 'DataFrame') -> 'DataFrame':
        assert len(self._columns) == len(other._columns)
        result_dict: dict = {}
        # data = [{name: col[i] for name, col in self._columns.items()} for i in range(len(self))]
        # data_other = [{name: col[i] for name, col in other._columns.items()} for i in range(len(other))]

        for name in self._columns.keys():
            result_col_values_self: list = []
            result_col_values_other: list = []
            for i in range(self._size):
                if not (self._columns[name][i] == other._columns[name][i]):
                    result_col_values_self.append(self._columns[name][i])
                    result_col_values_other.append(other._columns[name][i])
                else:
                    result_col_values_self.append(None)
                    result_col_values_other.append(None)
            result_col_self = Column(result_col_values_self, self._columns[name].dtype)
            result_col_other = Column(result_col_values_other, other._columns[name].dtype)
            result_dict.update({name: result_col_self})
            result_dict.update({name+"_other": result_col_other})
        return DataFrame(result_dict)

    def sample(self, number_rows: int) -> 'DataFrame':
        indices_of_rows = random.sample(range(0, self._size), number_rows)
        data = [{name: col[i] for name, col in self._columns.items()} for i in range(self._size)]
        data_sample: list[dict] = []
        for index in sorted(indices_of_rows):
            data_sample.append(data[index])
        result_dict: dict = {name: [row[name] for row in data_sample] for name in self.columns}
        for ele in result_dict:
            if type(result_dict[ele][0]) == float:
                result_dict[ele] = Column(result_dict[ele], Type.Float)
            else:
                result_dict[ele] = Column(result_dict[ele], Type.String)
        return DataFrame(result_dict)

    def melt(self, id_vars: list, value_vars: list) -> 'DataFrame':
        result_dict: dict = {}
        id_vars_x: dict = {name: [] for name in id_vars}
        variable: list = []
        values: list = []

        for col_name in value_vars:
            for id_name in id_vars_x:
                id_vars_x[id_name].extend(self._columns[id_name][::])
            for i in range(self._size):
                variable.append(col_name)
            values.extend(self._columns[col_name][::])
        for id in id_vars:
            result_dict.update({id: Column(id_vars_x[id], Type.String)})
        result_dict.update({'variable': Column(variable, Type.String)})
        result_dict.update({'values': Column(values, Type.String)})

        return DataFrame(result_dict)
        # Kinda did it but everything is cast to Type.String (annoying)

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
        if len(row) != len(self._columns):
            print(f"Wrong size: length of dataframe - {len(self._columns)}, length of given row - {len(row)}")
            return 0
        index: int = 0
        for name, col in self._columns.items():
            col.append(row[index])  # append method handles ValueError, because it utilizes _cast attribute ->
            index += 1              # therefore no need to implement try/except. i.e. "7" is cast to 7, "a" is ValueError
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
        """Couldn't do it myself
            src: github.com/ValdemarPospisil/Megatherion/blob/main/megatherion.py#L197
        """
        # Mainly just this part was ctrl-cv'ed
        data = [{name: col[i] for name, col in self._columns.items()} for i in range(len(self))]
        sorted_data = self.__merge_sort(data, col_name, ascending)
        sorted_columns = {name: [row[name] for row in sorted_data] for name in self.columns}
        # ---
        # Did only this myself
        for ele in sorted_columns:
            if type(sorted_columns[ele][0]) == float:
                sorted_columns[ele] = Column(sorted_columns[ele], Type.Float)
            else:
                sorted_columns[ele] = Column(sorted_columns[ele], Type.String)
        # ---
        return DataFrame(sorted_columns)

    @staticmethod
    def __merge_sort(arr, col_name, ascending=True):
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left = DataFrame.__merge_sort(arr[:mid], col_name, ascending)
        right = DataFrame.__merge_sort(arr[mid:], col_name, ascending)

        return DataFrame.__merge(left, right, col_name, ascending)

    @staticmethod
    def __merge(left, right, col_name, ascending):
        result = []
        left_idx, right_idx = 0, 0

        while left_idx < len(left) and right_idx < len(right):
            if (left[left_idx][col_name] < right[right_idx][col_name]) if ascending else (
                    left[left_idx][col_name] > right[right_idx][col_name]):
                result.append(left[left_idx])
                left_idx += 1
            else:
                result.append(right[right_idx])
                right_idx += 1

        result.extend(left[left_idx:])
        result.extend(right[right_idx:])
        return result

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
        #Reseni
        """it does in fact work"""
        result_df = copy.deepcopy(self)
        keys: list = []
        for ele in self._columns[self_key_column]:
            for i in range(len(other._columns[other_key_column])):
                if other._columns[other_key_column][i] == ele:
                    keys.append(i)
        #print(f"keys: {keys}")
        for name, col in other._columns.items():
            if name == other_key_column:
                continue
            new_col_values: list = []
            for key in keys:
                new_col_values.append(col[key])
            result_df.append_column(name, Column(new_col_values, col.dtype))
        return result_df

    def setvalue(self, col_name: str, row_index: int, value: Any) -> None:
        """
        Set new value in dataframe.
        :param col_name:  name of columns
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
