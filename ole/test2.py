import olefile

def parse_ole_file(file_path):
    # 打开 OLE 文件
    ole = olefile.OleFileIO(file_path)

    # 检查文件是否为 OLE 文件
    if ole.exists('Encrypted'):
        print("The file is encrypted.")
    else:
        print("The file is not encrypted.")

    # 列出文件中的所有存储流
    print("Streams in the OLE file:")
    for stream in ole.listdir():
        print(stream)

    # 获取某个流的内容
    if ole.exists('Workbook'):
        workbook_stream = ole.openstream('Workbook')
        print("Content of Workbook stream:")
        print(workbook_stream.read(100))  # 读取前100个字节

# 示例：解析一个 OLE 文件（如 .xls 文件）
file_path = '111111.xls'
parse_ole_file(file_path)

print("===========")

file_path = '222222.xls'
parse_ole_file(file_path)
