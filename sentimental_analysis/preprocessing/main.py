if __name__ == "__main__":
    import os

    if os.path.exists("data/preprocessed_data.csv"):
        print("Data already preprocessed")
        if os.path.exists("data/balanced_data.csv"):
            print("Data already balanced")
        else:
            import balance
    else:
        import preprocess
        import balance