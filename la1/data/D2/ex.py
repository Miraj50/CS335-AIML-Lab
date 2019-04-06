from collections import Counter, defaultdict

hola = [i*5 for i in range(1, 41)]
d = defaultdict(list)
for temp in hola:
	with open("test_data", "r") as f, open("test_labels", "r") as f1:
		for _ in range(temp):#f:
			a = "".join(next(f).strip().split())
			c=0
			for i in range(1,49):
				for j in range(1,49):
					p = a[i*50+j]
					if p == '0':
						if a[i*50+j+1] == '1' or a[i*50+j-1] == '1' or a[i*50+50+j] == '1':# or a[i*50-50+j] == '1' or a[(i-1)*50+j+1] == '1' or a[(i-1)*50+j-1] == '1' or a[(i+1)*50+j-1] == '1' or a[(i+1)*50+j+1] == '1':
							c+=1

			tt = next(f1).strip()
			c1 = Counter(a)
			# d[int(tt)].append(c1['1'])
			d[int(tt)].append(c1['1']/c)
		print(temp, {i:sum(d[i])/len(d[i]) for i in d})
		# print(f"{tt, c}")
		# c = Counter(a)
		# d[int(next(f1).strip())].append(c['0']/(2500))
	# print(d)
	# print({i:sum(d[i])/len(d[i]) for i in d})
	# for line in f:
	# 	a = "".join(line.strip().split())
	# 	print(a, len(a))
	# 	break