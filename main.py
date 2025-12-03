import os

def count_lines_in_directory_py(directory_path):
    total_lines = 0
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    total_lines += count_lines_in_file(file_path)
                except Exception as e:
                    print(f"Could not read file {file_path}: {e}")
    return total_lines

def count_lines_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return sum(1 for line in file)

def count_lines_in_directory(directory_path):
    total_lines = 0
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_lines += count_lines_in_file(file_path)
            except Exception as e:
                print(f"Could not read file {file_path}: {e}")
    return total_lines

if __name__ == "__main__":
    directory_path = './'
    total_lines = count_lines_in_directory_py(directory_path)
    print(f"py文件总共有:{total_lines}行")
