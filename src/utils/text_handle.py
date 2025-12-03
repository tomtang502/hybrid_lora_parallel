import os

def create_txt_file(file_path):
    """
    Creates a new text file at the specified path. 
    If the file exists, it empties/overwrites it.
    """
    if os.path.dirname(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
    with open(file_path, 'w') as f:
        pass

def append_to_txt_file(file_path, content):
    """
    Appends the provided content string to the end of the specified file.
    Adds a newline character at the end of the content.
    """
    with open(file_path, 'a') as f:
        f.write(content + '\n')