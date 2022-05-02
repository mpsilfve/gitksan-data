from os import makedirs

def map_list(f, l):
    return list(map(f, l))

def try_makedirs(folder):
    try:
        makedirs(folder)
    except FileExistsError:
        pass
    

