import os

def check(path):
    for item in os.scandir(path):
        if item.is_file() and not item.name.endswith(".gz"):
            print(item.name)
        if item.is_dir():
            check(item.path)

if __name__ == "__main__":
    check(".")
