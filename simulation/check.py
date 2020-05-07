import sys

if __name__ == '__main__':
	file1 = sys.argv[1]
	file2 = sys.argv[2]
	with open(file1, 'r') as f1:
		with open(file2, 'r') as f2:
			for l1, l2 in zip(f1, f2):
				if l1 != l2:
					print("false")
					print(l1)
					print(l2)
					break
	print("DONE")