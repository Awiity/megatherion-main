from megatherion import DataFrame, Column, Type

#col1 = Column([1, 2, 3, 4, 5, 6], Type.Float)
#col2 = Column([1, 1, 2, 3, 5, 8], Type.Float)
#col3 = Column([1, 4, 9, 16, 25, 36], Type.Float)
col1 = Column([2, 3, 1, 0], Type.Float)
col2 = Column([1, None, 0,0], Type.Float)
col3 = Column(["Niga", "whatever", "A", "B"], Type.String)
df = DataFrame(dict(c1=col1, c2=col2, c3=col3))
"""TESTING HERE"""

#print(df)
df.append_row(row=(99.0, 99.0, "ficl"))
df_c = df.sort("c1", True)

print(df.describe())


#   MEMORIZE IT
for name, data in df._columns.items():
    print(name, data._data)