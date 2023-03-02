
from cProfile import label
from inspect import ismemberdescriptor
import logging
import numpy as np
import torch
import pickle as pkl

import cv2
import os
import time
from sklearn.preprocessing import MultiLabelBinarizer
logger = logging.getLogger(__name__)

def extract_frames(video_path,  overwrite=False, start=-1, end=-1, every=1):
    """
    Extract frames from a video using OpenCVs VideoCapture
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """
    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible

    assert os.path.exists(video_path)  # assert the video file exists
    capture = cv2.VideoCapture(video_path)  # open the video using OpenCV
    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved
    img_list=[]
    #print(start)
    while frame < end:  # lets loop through the frames until the end
        _, image = capture.read()  # read an image from the capture
        if while_safety > 500:  # break the while if our safety maxs out at 500
            break
        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            print('false frame')
            continue  # skip
        if frame % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            #save_path = os.path.join(frames_dir, video_filename, "{:010d}.jpg".format(frame))  # create the save path
            #if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
            #    cv2.imwrite(save_path, image)  # save the extracted image
            #    saved_count += 1  # increment our counter by one
            image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #print(image.shape)
            img_list.append(cv2.resize(image, (224,224)))
            #if (videop == 'vp14/run1_2018-05-30-10-11-09.ids_1.avi') and (start == 10486):
            #print(image)
        frame += 1  # increment our frame count
    capture.release()  # after the while has finished close the capture
    return np.array(img_list)  # and return the count of the images we saved

class MuscleMapV2(torch.utils.data.Dataset):
    """
    AVA Dataset
    """
    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = cfg.MODEL.NUM_CLASSES
        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.AVA.BGR
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP
        self.clip_num=16
        if self._split == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.AVA.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.AVA.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.DATA.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.DATA.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE
            self._test_force_flip = cfg.AVA.TEST_FORCE_FLIP
        if self._split == 'train':
            #f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_allseen/train_split_muscle_map.pkl','rb')
            f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/train_split2_muscle_map.pkl','rb')
            self.video_list= pkl.load(f)
            f.close()
            self.labels = [lookup_dict[item.split('AN_')[-1].split('_ST')[0]] for item in self.video_list]
            f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/train_split2_muscle_map_label.pkl','wb')
            pkl.dump(obj=self.labels, file=f)
            f.close()
            #f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_withunseen/train_split2_muscle_map_label.pkl','rb')
            #f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_allseen/train_split_muscle_map_label.pkl','rb')
            #self.labels= pkl.load(f)
            #f.close()
        elif self._split == 'val_seen':
            #f=open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_allseen/val_split_muscle_map.pkl','rb')
            f=open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/val_seen_split2_muscle_map.pkl','rb')
            self.video_list= pkl.load(f)
            f.close()
            self.labels = [lookup_dict[item.split('AN_')[-1].split('_ST')[0]] for item in self.video_list]
            f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/val_seen_split2_muscle_map_label.pkl','wb')
            pkl.dump(obj=self.labels, file=f)
            f.close()
        elif self._split == 'val_unseen':
            #f=open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_allseen/val_split_muscle_map.pkl','rb')
            f=open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/val_unseen_split2_muscle_map.pkl','rb')
            self.video_list= pkl.load(f)
            f.close()
            self.labels = [lookup_dict[item.split('AN_')[-1].split('_ST')[0]] for item in self.video_list]
            f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/val_unseen_split2_muscle_map_label.pkl','wb')
            pkl.dump(obj=self.labels, file=f)
            f.close()
            #f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_allseen/val_split_muscle_map_label.pkl','rb')

            #f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_withunseen/val_split2_muscle_map_label.pkl','rb')
            #self.labels= pkl.load(f)
            #f.close()
        elif self._split == 'test_seen':
            #f=open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_allseen/test_split_muscle_map.pkl','rb')
            f=open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/test_seen_split2_muscle_map.pkl','rb')
            self.video_list= pkl.load(f)
            f.close()
            self.labels = [lookup_dict[item.split('AN_')[-1].split('_ST')[0]] for item in self.video_list]
            f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/test_seen_split2_muscle_map_label.pkl','wb')
            pkl.dump(obj=self.labels, file=f)
            f.close()
        else:
            #f=open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_allseen/test_split_muscle_map.pkl','rb')
            f=open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/test_unseen_split2_muscle_map.pkl','rb')
            self.video_list= pkl.load(f)
            f.close()
            self.labels = [lookup_dict[item.split('AN_')[-1].split('_ST')[0]] for item in self.video_list]
            f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/test_unseen_split2_muscle_map_label.pkl','wb')
            pkl.dump(obj=self.labels, file=f)
            f.close()
            #f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_withunseen/test_split2_muscle_map_label.pkl','rb')
            #f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_allseen/test_split_muscle_map_label.pkl','rb')
            #self.labels= pkl.load(f)
            #f.close()
        one_hot = MultiLabelBinarizer()
        self.original_labels = self.labels
        self.labels.append([item for item in range(20)])
        self.labels = one_hot.fit_transform(self.labels)
        self.labels = self.labels[:-1]
        f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/muscle_map_annotation_embedding.pkl', 'rb')
        self.semantic_embeddings = pkl.load(f)
        f.close()
    def get_semantic_embedding_and_edge_index(self,labels):
        adj = torch.ones(20,20)
        for i in range(20):
            if i not in labels:
                adj[i,:]=0
                adj[:,i]=0
        return adj, torch.Tensor(self.semantic_embeddings[:20])
    def clip_generation(self, video):
        stride = video.shape[0]//self.clip_num
        canvas = np.zeros([self.clip_num, video.shape[1], video.shape[2], video.shape[3]])
        if stride == 0:
            repeated = np.concatenate([video]*20, axis=0)
            canvas = video[:self.clip_num]
        else:
            unit = video[::stride,...]
            repeated = np.concatenate([unit]*20, axis=0)
            canvas =repeated[:self.clip_num]
        return canvas     
    def print_summary(self):
        logger.info("=== MuscleMAP dataset summary ===")
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(len(self._image_paths)))
    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self.video_list)

    def _images_preprocessing(self, imgs):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs = imgs / 255.0
        # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
        # range of [0, 1].

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = transform.color_jitter(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )
            imgs = transform.lighting_jitter(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = transform.color_normalization(
            imgs,
            np.array(self._data_mean, dtype=np.float32),
            np.array(self._data_std, dtype=np.float32),
        )
        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            # Note that Kinetics pre-training uses RGB!
            imgs = imgs[:, [2, 1, 0], ...]

        return imgs
    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            time index (zero): The time index is currently not supported for AVA.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        start = time.time()
        # When short cycle is used, input index is a tupple.
        path = self.video_list[idx]
        path = path.replace('musclemap_trimmed','musclemap_rescale')
        
        while os.path.exists(os.path.normpath(path)) == False:
            import random
            idx = random.choice(range(0,len(self.video_list)))
            path = self.video_list[idx]
            path = path.replace('musclemap_trimmed','musclemap_rescale')
        imgs = extract_frames(path)
        
        while len(imgs.shape) != 4:
            import random
            idx = random.choice(range(0,len(self.video_list)))
            path = self.video_list[idx]
            path = path.replace('musclemap_trimmed','musclemap_rescale')
            imgs = extract_frames(self.video_list[idx])
        #adj, embeddings = self.get_semantic_embedding_and_edge_index(self.original_labels[idx])
        #print(embeddings.size())
        imgs = torch.Tensor(self.clip_generation(imgs))
        # T H W C -> T C H W.
        imgs = imgs.permute(0, 3, 1, 2)
        # Preprocess images and boxes.
        imgs = self._images_preprocessing(imgs)
        # T C H W -> C T H W.
        imgs = imgs.permute(1, 0, 2, 3)
        # Construct label arrays.
        imgs = utils.pack_pathway_output(self.cfg, imgs)
        annotation = self.labels[idx]
        #end = time.time()
        #print(end-start)
        imgs = imgs[0]
        C,T,H,W = imgs.size()
        if T > 16:
            imgs = imgs[:16]
        if T < 16:
            imgs = torch.cat([imgs, torch.zeros([C, 16-T,H,W])], dim=1)
        #print('testssssssssssssssssssssssssssssssssssssssssssssssssss')
        if self._split == 'train':
            #print('test')
            return [imgs,], [torch.Tensor(annotation)], torch.Tensor(idx), [torch.zeros(1)], {}
        else:
            return [imgs,], torch.Tensor(annotation), idx, torch.zeros(1), {}


if __name__ == "__main__":
    for _split in ['train','val_seen', 'val_unseen', 'test_seen', 'test_unseen']:
        if _split == 'train':
            f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/train_split2_muscle_map.pkl','rb')
            video_list= pkl.load(f)
            f.close()
            name_video = '/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/train_split2_muscle_map_cleanup.pkl'

            f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/train_split2_muscle_map_label.pkl','rb')
            labels= pkl.load(f)
            f.close()

            name_label = '/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/train_split2_muscle_map_label_cleanup.pkl'
        elif _split == 'val_seen':
            f=open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/val_seen_split2_muscle_map.pkl','rb')
            video_list= pkl.load(f)
            f.close()
            name_video = '/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/val_seen_split2_muscle_map_cleanup.pkl'


            f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/val_seen_split2_muscle_map_label.pkl','rb')
            labels= pkl.load(f)
            f.close()

            name_label = '/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/val_seen_split2_muscle_map_label_cleanup.pkl'


        elif _split == 'val_unseen':
            f=open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/val_unseen_split2_muscle_map.pkl','rb')
            video_list= pkl.load(f)
            f.close()
            name_video = '/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/val_unseen_split2_muscle_map_cleanup.pkl'


            f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/val_unseen_split2_muscle_map_label.pkl','rb')
            labels= pkl.load(f)
            f.close()

            name_label = '/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/val_unseen_split2_muscle_map_label_cleanup.pkl'
        elif _split == 'test_seen':
            f=open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/test_seen_split2_muscle_map.pkl','rb')
            video_list= pkl.load(f)
            f.close()
            name_video = '/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/test_seen_split2_muscle_map_cleanup.pkl'


            f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/test_seen_split2_muscle_map_label.pkl','rb')
            labels= pkl.load(f)
            f.close()

            name_label = '/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/test_seen_split2_muscle_map_label_cleanup.pkl'
        else:
            f=open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/test_unseen_split2_muscle_map.pkl','rb')
            video_list= pkl.load(f)
            f.close()
            name_video = '/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/test_unseen_split2_muscle_map_cleanup.pkl'

            f = open('/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/test_unseen_split2_muscle_map_label.pkl','rb')
            labels= pkl.load(f)
            f.close()

            name_label = '/home/haicore-project-kit-iar-cvhci/fy2374/cvpr2023/train_val_test_musclemap_mixed_all/test_unseen_split2_muscle_map_label_cleanup.pkl'
        video_list_copy = video_list
        clean_list = []
        clean_label = []
        for idx in range(len(video_list)):

            path = video_list[idx]
            path_x = path.replace('musclemap_trimmed','musclemap_rescale')
            
            if (os.path.exists(os.path.normpath(path)) == False) or (os.path.exists(os.path.normpath(path_x)) == False):
                del_index = video_list.index(path)
                #del video_list_copy[del_index]
                #del labels[del_index]
                continue
            imgs = extract_frames(path_x)
            if len(imgs.shape) != 4:
                #del_index = video_list.index(path)
                #del video_list_copy[del_index]
                #del labels[del_index]
                continue
            clean_list.append(path)
            clean_label.append(labels[idx])            
        f = open(name_video, 'wb+')
        pkl.dump(file=f, obj=clean_list)
        f.close()
        f = open(name_label, 'wb+')
        pkl.dump(file=f, obj=clean_label)
        f.close()
