# Let's write code to compare two text files line by line and check if they are identical.
# We'll assume filenames for now; user will provide them or we can generalize.

file1 = "./log_new.txt"
file2 = "./log_old.txt"

def compare_files(f1, f2):
    diffs = []
    with open(f1, "r", encoding="utf-8", errors="ignore") as a, open(f2, "r", encoding="utf-8", errors="ignore") as b:
        for i, (line1, line2) in enumerate(zip(a, b), start=1):
            if line1 != line2:
                diffs.append((i, line1.strip(), line2.strip()))
        # Check if one file has extra lines
        for i, line1 in enumerate(a, start=i+1):
            diffs.append((i, line1.strip(), "<EOF>"))
        for i, line2 in enumerate(b, start=i+1):
            diffs.append((i, "<EOF>", line2.strip()))
    return diffs

diffs = compare_files(file1, file2)
len(diffs), diffs[:10]

if __name__ == "__main__":
    if diffs:
        print(f"Files differ at {len(diffs)} lines:")
        for line_num, line1, line2 in diffs:
            print(f"Line {line_num}:")
            print(f"  File New: {line1}")
            print(f"  File Old: {line2}")
    else:
        print("Files are identical.")