import sys
import h5py
import numpy as np
from timeit import default_timer as timer


def location(video_id):
    if video_id in [1, 2, 3, 4]:
        loc = 1
    elif video_id in [5, 6, 7, 8, 9, 10]:
        loc = 2
    elif video_id in [11, 12]:
        loc = 3
    elif video_id in [13, 14, 15]:
        loc = 4
    else:
        raise ValueError()
    return loc


def generateScoresWithGroups():

    vals, ids = [], []

    assigned_ids = set()

    all_locations = [location(k) for k in x_desc[:, 0]]
    all_locations_set = set(all_locations)

    n_examples = x.shape[0]
    for index in range(0, n_examples):

        start_time = timer()

        if y_id[index] not in assigned_ids:

            current_video_loc = all_locations[index]

            current_example_id = y_id[index]
            assigned_ids.add(current_example_id)

            other_video_locs = list(all_locations_set - set([current_video_loc]))

            current_vals, current_ids = [], []

            for video_loc in other_video_locs:

                condition = np.asarray(all_locations) == video_loc

                x_in_video_loc = x[condition]
                y_in_video_loc = y_id[condition]

                dist_with_all_in_video_loc = np.sqrt(np.sum((x[index] - x_in_video_loc) ** 2, axis=-1))

                min_d = np.min(dist_with_all_in_video_loc)
                argmin_d = np.argmin(dist_with_all_in_video_loc)
                y_min_dist = y_in_video_loc[argmin_d]
                if (min_d < thr) and (y_min_dist not in assigned_ids):
                    if not current_ids:
                        current_ids.append(current_example_id)
                    current_vals.append(min_d)
                    current_ids.append(y_min_dist)
                    assigned_ids.add(y_min_dist)

            vals.append(current_vals)
            ids.append(current_ids)

        elapsed = timer() - start_time
        print('{} / {}  -  Time for this iteration: {}s'.format(index, n_examples, elapsed))

    average_score = [np.mean(v) for v in vals]

    # Filter average score to remove the NaNs
    average_score = np.asarray(average_score)
    ids = np.asarray(ids)

    idxs_to_keep = True ^ np.isnan(average_score)

    average_score = average_score[idxs_to_keep]
    ids = ids[idxs_to_keep]

    sort_indexes = np.argsort(average_score)
    sorted_ids = ids[sort_indexes]
    sorted_score = average_score[sort_indexes]

    # Keep only cars that appear in all 4 locations
    all_4_locs_idxs = [i for (i, l) in enumerate(sorted_ids) if len(l) == 4]
    candidates = sorted_ids[all_4_locs_idxs]
    sorted_score = sorted_score[all_4_locs_idxs]

    max_score = np.max(sorted_score)

    normalized_score = 1. - sorted_score / max_score
    normalized_score /= np.max(normalized_score)

    return candidates, normalized_score


if __name__ == '__main__':

    miniDataset_path = sys.argv[1]
    out_file_path = sys.argv[2]
    thr = int(sys.argv[3])


    h5train = h5py.File(miniDataset_path, 'r')
    x       = np.asarray(h5train['X'])
    y_id    = np.asarray(h5train['Y_ID'])
    x_desc  = np.asarray(h5train['desc'])

    candidates, normalized_score = generateScoresWithGroups()

    # Dump output
    with open(out_file_path, 'w') as f:
        for score, candidate in zip(normalized_score, candidates):
            line = '{} {} {} {} {}\n'.format(score, *candidate)
            f.write(line)