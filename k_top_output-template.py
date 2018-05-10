__author__ = 'pedro'

import sys
import h5py

db_path = sys.argv[1]
fich_groups_path = sys.argv[2]
out_fich_path = sys.argv[3]

h5 = h5py.File(db_path, 'r')
X = h5['X']
Y_id = h5['Y_ID']
Y_desc = h5['desc']

out = open(out_fich_path, 'w')

for index in range(len(Y_id)):
    groups = open(fich_groups_path, 'r')
    idd_pos = 0
    isFound = False
    for line in groups:
        idd_pos += 1
        if idd_pos > 100:
            groups.close()
            break
        idsToMerge = line.strip().split(' ')
        for term in idsToMerge:
            if idsToMerge.index(term) == 0:
                conf = '<' + str(term) + '>'
            else:
                if int(term) == Y_id[index]:
                    #<video_id> <frame_id> <obj_id> <xmin> <ymin> <xmax> <ymax> <speed> <confidence>
                    videoId = '<' + str(Y_desc[index,0]) + '> '
                    frameId = '<' + str(int(Y_desc[index,1])) + '> ' #rango de 1..n
                    objId = '<' + str(idd_pos) + '> '
                    xmin = '<' + str(Y_desc[index, 4]) + '> '
                    ymin = '<' + str(Y_desc[index, 2]) + '> '
                    xmax = '<' + str(Y_desc[index, 5]) + '> '
                    ymax = '<' + str(Y_desc[index, 3]) + '> '
                    out.write(videoId + frameId + objId + xmin + ymin + xmax + ymax + ' <-1> ' + conf+'\n')
                    isFound = True
                    break
        if isFound:
            break
    groups.close()
out.close()