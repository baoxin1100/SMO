def load_data_set(fileName):
	data_mat = []; label_mat = []
	fr = open(fileName)
	for line in fr:
		line_arr = line.strip().split('\t')
		data_mat.append([float(line_arr[0]), float(line_arr[1])])
		label_mat.append(float(line_arr[2]))
	return data_mat, label_mat

def select_jrand(i, m):
	j=i
	while (j==i):
		j = int(random.uniform(0,m))
	return j

def clip_alpha(aj, H, L):
	if aj > H:
		aj = H
	if L > aj:
		aj = L
	return aj