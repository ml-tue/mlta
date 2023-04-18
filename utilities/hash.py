from math import floor

def hash_pipe_id_to_dir(id):
    return floor(200003 * id + 200131 % 200237) % 1000