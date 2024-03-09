import os
import glob
import cv2

# input_folder = 'trainA'

# input_folder = 'valA'

input_folder = 'testA'


frame_sparsity = 3

overlap = 0


def sparse_im(imgs_path, sparse_path, overlap):
    if input_folder == 'testA':
        im_files = sorted(list(glob.glob(os.path.join(imgs_path, '*.jpg'))))
    else:
        im_files = sorted(list(glob.glob(os.path.join(imgs_path, '*.tif'))))

    for k, im_file0 in enumerate(im_files):

        if frame_sparsity > 1 and k % frame_sparsity != 0 or (frame_sparsity == 3 and k == len(im_files)-1):
            continue
        im_name = os.path.split(im_file0)[-1].split('.')[0]
        im = cv2.imread(im_file0)

        if not os.path.exists(sparse_path):
            os.makedirs(sparse_path)

        # new_img_path = os.path.join(sparse_path, ('%08d' % (k // frame_sparsity) + '.tif'))
        new_img_path = os.path.join(sparse_path, ('%08d' % k + '.tif'))
        cv2.imwrite(new_img_path, im)


if input_folder == 'testA':
    im_folders = os.listdir('datasets/ben_frames/{}'.format(input_folder))
else:
    im_folders = os.listdir('datasets/ben_frames/splitting_4/{}'.format(input_folder))
for k, folder in enumerate(im_folders):
    print(folder)
    if input_folder == 'testA':
        input_imgs_path = 'datasets/ben_frames/{}/{}'.format(input_folder, folder)
    else:
        input_imgs_path = 'datasets/ben_frames/splitting_4/{}/{}'.format(input_folder, folder)
    input_sparse_path = 'datasets/ben_frames/temporal_sparsity/sparsity_{}/{}/{}'.format(frame_sparsity, input_folder, folder)

    sparse_im(input_imgs_path, input_sparse_path, overlap)

print('done!')
