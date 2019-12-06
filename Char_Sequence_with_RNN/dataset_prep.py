import os


def find_files(suffix, path):
    """
    Recursively creates a list of all the files under the directory and subdirectories

    :arguments:
    ----------------------------------------
        suffix  :   File extension
        path    :   Location of the directory
    :return:
     ---------------------------------------
        arr     :   list of all the file paths matching the ext

    """

    arr = list()

    # If no more dir available return empty list
    if path.strip(' ') == "":
        return []
    else:
        # Get the list of dirs
        paths = os.listdir(path)
        parent = path

        if paths is None or len(paths) == 0:
            return []
        else:
            # Loop through all the sub dirs.

            for p in paths:

                # In case the entry is a file, append the full path to the list
                if os.path.isfile(os.path.join(parent, p)) and p.endswith(suffix):
                    arr.append(os.path.join(parent, p))

                # In case the entry is a dir, call the find_files() function recursively
                if os.path.isdir(os.path.join(parent, p)):
                    child_arr = find_files(suffix, os.path.join(parent, p))
                    arr.extend(child_arr)

    return arr


if __name__ == '__main__':

    # Get the list of files with ext as .py
    file_list = find_files(".py", "/Users/home/Downloads/scikit-learn-master")

    new_file = ''

    # Loop thorugh the list
    for file in file_list:
        with open(file, 'r') as f:
            new_file += '<start>'
            new_file += f.read()
            new_file += '<end>'

    # Save as a new file
    with open("datasets/data.txt", 'w+') as f:
        f.write(new_file)
