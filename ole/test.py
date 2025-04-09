import olefile

def is_encrypted(file_path):
    ole = olefile.OleFileIO(file_path)
    if ole.exists('Encrypted'):
        return True
    return False

print(is_encrypted('111111.xls'))
print(is_encrypted('222222.xls'))

