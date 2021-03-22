from jaad_data import JAAD
import pickle

jaad_path = '/home/romaincaristan/data/romaincaristan-data/JAAD/'
imdb = JAAD(data_path=jaad_path)

imdb.generate_database()

data = pickle.load(open(jaad_path+'data_cache/jaad_database.pkl', 'rb'))


file_out = open("results_new.csv", "w")

for videos in data.keys():
	ped_anno = data[videos]["ped_annotations"]
	for d in ped_anno:
		if 'look' in ped_anno[d]['behavior'].keys():
			for i in range(len(ped_anno[d]['frames'])):
				line = ','.join([videos, d, str(ped_anno[d]['frames'][i]), str(ped_anno[d]['bbox'][i][0]), str(ped_anno[d]['bbox'][i][1]), str(ped_anno[d]['bbox'][i][2]), str(ped_anno[d]['bbox'][i][3]), str(ped_anno[d]['behavior']['look'][i])])
				file_out.write(line+'\n')

file_out.close()
