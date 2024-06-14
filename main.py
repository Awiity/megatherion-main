from megatherion import DataFrame, Column, Type

col1 = Column([1, 2, 3, 4, 5, 6], Type.Float)
col2 = Column(["M&M", "LAYS", "COLA", "FR_S", "MARS", "PPSI"], Type.String)
col3 = Column([66, 45, 87, 44, 678, 98], Type.Float)
col4 = Column([2, 1, 3, 2, 2, 3], Type.Float)

df = DataFrame(dict(prod_id=col1, prod_name=col2, stock=col3, category=col4))

col1 = Column([1, 2, 3], Type.Float)
col2 = Column(["Salty", "Sweet", "Soda"], Type.String)
col3 = Column(["Sodium-rich product", "Sugar-rich shit", "Liquid shit that kills you"], Type.String)

other_df = DataFrame(dict(category_id=col1, cat_name=col2, desc=col3))

col1 = Column([1, 2, 3, 4, 5, 6], Type.Float)
col2 = Column([1, 1, 2, 3, 5, 8], Type.Float)
col3 = Column([1, 4, 9, 16, 25, 36], Type.Float)
col4 = Column([1, 1, 1, 1, 1, 1], Type.Float)

df_num = DataFrame(dict(c1=col1, c2=col2, c3=col3, c4=col4))

col1 = Column([5, 1, 4, 2, 3, 6], Type.Float)
col2 = Column(["tank", "rifle", "pistol", "cannon", "knife", "jet"], Type.String)
col3 = Column([1000000, 1000, 500, 3000, 100, 75000000], Type.Float)

df_sort = DataFrame(dict(id=col1, name=col2, price=col3))

col1 = Column(['a', 'a', 'b', 'b', 'a'], Type.String)
col2 = Column([1, 2, 3, None, 5], Type.Float)
col3 = Column([1, 2, 3, 4, 5], Type.Float)

df_1 = DataFrame(dict(col1=col1, col2=col2, col3=col3))

col1 = Column(['c', 'a', 'b', 'b', 'a'], Type.String)
col2 = Column([1, 2, 3, None, 5], Type.Float)
col3 = Column([1, 2, 4, 4, 5], Type.Float)

df_2 = DataFrame(dict(col1=col1, col2=col2, col3=col3))

col1 = Column([1, 4], Type.Float)
col2 = Column([2, 5], Type.Float)
col3 = Column([3, 6], Type.Float)

df_dot = DataFrame(dict(c1=col1, c2=col2, c3=col3))

col1 = Column([1, 3, 5], Type.Float)
col2 = Column([2, 4, 6], Type.Float)


df_dot2 = DataFrame(dict(c1=col1, c2=col2))

col1 = Column(['a', 'b', 'c'], Type.String)
col2 = Column([1, 3, 5], Type.Float)
col3 = Column([2, 4, 6], Type.Float)

df_melt = DataFrame(dict(A=col1, B=col2, C=col3))
"""TESTING HERE"""

#print(df_melt)
print(df_melt.melt(['A'], ['B','C']))

#print(df_dot.dot(df_dot2))

#df_num.diff(axis=1)
#print(df_num)

#print(df_num.sample(3))

#print(df_num.cumsum(axis=1))
#df_1.compare(df_2)

# dat = [{name: col[i] for name, col in df_sort._columns.items()} for i in range(len(df_sort))]
# sorted_columns = {name: [ele[name] for ele in dat] for name in df_sort.columns}
#
# for ele in sorted_columns:
#     if type(sorted_columns[ele][0]) == float:
#         sorted_columns[ele] = Column(sorted_columns[ele], Type.Float)
#     else:
#         sorted_columns[ele] = Column(sorted_columns[ele], Type.String)

# df_sorted = df_sort.sort("id")
#
# print(df_sorted)
