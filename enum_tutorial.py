from megatherion import DataFrame, Type, Column

if __name__ == "__main__":
    df = DataFrame(dict(
        a=Column([None, 3.1415], Type.Float),
        b=Column(["a", 2], Type.String),
        c=Column(range(2), Type.Float)
    ))
    df.setvalue("a", 1, 42)
    print(df)

    df = DataFrame.read_json("data.json")
    print(df)