
import numpy as np
import os
import random
import pandas
import torch
import torch.utils.data
from torchvision import transforms

import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr

from . import decoder as decoder
from . import transform as transform
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from .random_erasing import RandomErasing
from .transform import create_random_augment
import pickle as pkl
logger = logging.get_logger(__name__)
from sklearn.preprocessing import MultiLabelBinarizer



@DATASET_REGISTRY.register()
class MuscleMap(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=16):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val_seen",
            "val_unseen",
            "test_seen",
            "test_unseen"
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.p_convert_gray = self.cfg.DATA.COLOR_RND_GRAYSCALE
        self.p_convert_dt = self.cfg.DATA.TIME_DIFF_PROB
        self._video_meta = {}
        self._num_retries = num_retries
        self._num_epoch = 0.0
        self._num_yielded = 0
        self.skip_rows = self.cfg.DATA.SKIP_ROWS
        self.use_chunk_loading = (
            True
            if self.mode in ["train"] and self.cfg.DATA.LOADER_CHUNK_SIZE > 0
            else False
        )
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val_seen", "val_unseen"]:
            self._num_clips = 1
        elif self.mode in ["test_seen", "test_unseen"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()
        self.randaug = False
        self.rand_erase = False
        self.use_temporal_gradient = False
        self.temporal_gradient_rate = 0.0

        if self.mode == "train" and self.cfg.AUG.ENABLE:
            self.randaug = True
            if self.cfg.AUG.RE_PROB > 0:
                self.rand_erase = True

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self.cur_iter = 0
        self.chunk_epoch = 0
        self.epoch = 0.0
        self.skip_rows = self.cfg.DATA.SKIP_ROWS
        if self.mode == "train":
            self._crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = self.cfg.AVA.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = self.cfg.AVA.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = self.cfg.DATA.TRAIN_PCA_EIGVAL
            self._pca_eigvec = self.cfg.DATA.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = self.cfg.DATA.TEST_CROP_SIZE
            self._test_force_flip = self.cfg.AVA.TEST_FORCE_FLIP
        if self.mode == 'train':
            f = open('path/to/train/video/list','rb')
            self._path_to_videos= pkl.load(f)
            f.close()
            f = open('path/to/train/label/list','rb')
            self._labels = pkl.load(f)
            f.close()
        elif self.mode == 'val_seen':
            f=open('path/to/val_seen/video/list','rb')
            self._path_to_videos= pkl.load(f)
            f.close()
            f = open('path/to/val_seen/label/list','rb')
            self._labels = pkl.load(f)
            f.close()
        elif self.mode == 'val_unseen':
            f=open('path/to/val_unseen/video/list','rb')
            self._path_to_videos= pkl.load(f)
            f.close()
            f = open('path/to/val_unseen/label/list','rb')
            self._labels = pkl.load(f)
            f.close()
        elif self.mode == 'test_seen':
            f=open('path/to/test_seen/video/list','rb')
            self._path_to_videos= pkl.load(f)
            f.close()
            f = open('path/to/test_seen/label/list','rb')
            self._labels = pkl.load(f)
            f.close()
        else:
            f=open('path/to/test_unseen/video/list','rb')
            self._path_to_videos= pkl.load(f)
            f.close()
            f = open('path/to/test_unseen/label/list','rb')
            self._labels = pkl.load(f)
            f.close()
        logger.info(
            "Constructing kinetics dataloader size: {}".format(
                len(self._path_to_videos)
            )
        )
        f = open('path/to/emd/list','rb') #not used
        self.embeddings = pkl.load(f)
        f.close()
        one_hot = MultiLabelBinarizer()
        self.original_labels = self._labels
        self._labels.append([item for item in range(20)])
        self._labels = one_hot.fit_transform(self._labels)
        self._labels = self._labels[:-1]
        self._spatial_temporal_idx = list(np.zeros(len(self._path_to_videos)))
        for i in range(len(self._path_to_videos)):
            self._video_meta[i] = {}

    def _set_epoch_num(self, epoch):
        self.epoch = epoch

    def _get_chunk(self, path_to_file, chunksize):
        try:
            for chunk in pandas.read_csv(
                path_to_file,
                chunksize=self.cfg.DATA.LOADER_CHUNK_SIZE,
                skiprows=self.skip_rows,
            ):
                break
        except Exception:
            self.skip_rows = 0
            return self._get_chunk(path_to_file, chunksize)
        else:
            return pandas.array(chunk.values.flatten(), dtype="string")

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, self._num_yielded = index
            if self.cfg.MULTIGRID.SHORT_CYCLE:
                index, short_cycle_idx = index

        if self.mode in ["train", "val_seen", "val_unseen"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test_seen", "test_unseen"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
                + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        num_decode = (
            self.cfg.DATA.TRAIN_CROP_NUM_TEMPORAL
            if self.mode in ["train"]
            else 1
        )
        min_scale, max_scale, crop_size = [min_scale], [max_scale], [crop_size]
        if len(min_scale) < num_decode:
            min_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * (
                num_decode - len(min_scale)
            )
            max_scale += [self.cfg.DATA.TRAIN_JITTER_SCALES[1]] * (
                num_decode - len(max_scale)
            )
            crop_size += (
                [self.cfg.MULTIGRID.DEFAULT_S] * (num_decode - len(crop_size))
                if self.cfg.MULTIGRID.LONG_CYCLE
                or self.cfg.MULTIGRID.SHORT_CYCLE
                else [self.cfg.DATA.TRAIN_CROP_SIZE]
                * (num_decode - len(crop_size))
            )
            assert self.mode in ["train", "val_seen", "val_unseen"]
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container_x = None
            #video_container_y = None
            video_path_x = self._path_to_videos[index].replace('musclemap_trimmed', 'musclemap_rescale')
            #print(video_path_x)
            #video_path_y = self._path_to_videos[index].replace('musclemap_trimmed', 'musclemap_opticalflow_video_y')
            
            video_container_x = container.get_video_container(
                video_path_x,
                self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                'pyav',
            )

            if video_container_x is None:
                logger.warning(
                    "Failed to meta load video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if self.mode not in ["test_seen", "test_unseen"] and i_try > self._num_retries // 8:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            frames_decoded, time_idx_decoded = (
                [None] * num_decode,
                [None] * num_decode,
            )

            # for i in range(num_decode):
            num_frames = [self.cfg.DATA.NUM_FRAMES]
            sampling_rate = utils.get_random_sampling_rate(
                self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
                self.cfg.DATA.SAMPLING_RATE,
            )
            sampling_rate = [sampling_rate]
            if len(num_frames) < num_decode:
                num_frames.extend(
                    [
                        num_frames[-1]
                        for i in range(num_decode - len(num_frames))
                    ]
                )
                # base case where keys have same frame-rate as query
                sampling_rate.extend(
                    [
                        sampling_rate[-1]
                        for i in range(num_decode - len(sampling_rate))
                    ]
                )
            elif len(num_frames) > num_decode:
                num_frames = num_frames[:num_decode]
                sampling_rate = sampling_rate[:num_decode]

            if self.mode in ["train"]:
                assert (
                    len(min_scale)
                    == len(max_scale)
                    == len(crop_size)
                    == num_decode
                )

            target_fps = self.cfg.DATA.TARGET_FPS
            if self.cfg.DATA.TRAIN_JITTER_FPS > 0.0 and self.mode in ["train"]:
                target_fps += random.uniform(
                    0.0, self.cfg.DATA.TRAIN_JITTER_FPS
                )

            # Decode video. Meta info is used to perform selective decoding.
            frames, time_idx, tdiff = decoder.decode(
                video_container_x,
                sampling_rate,
                num_frames,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index]
                if len(self._video_meta) < 5e6
                else {},  # do not cache on huge datasets
                target_fps=target_fps,
                backend='pyav',
                use_offset=self.cfg.DATA.USE_OFFSET_SAMPLING,
                max_spatial_scale=min_scale[0]
                if all(x == min_scale[0] for x in min_scale)
                else 0,  # if self.mode in ["test"] else 0,
                time_diff_prob=self.p_convert_dt
                if self.mode in ["train"]
                else 0.0,
                temporally_rnd_clips=True,
                min_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MIN,
                max_delta=self.cfg.CONTRASTIVE.DELTA_CLIPS_MAX,
            )
            frames_decoded = frames
            time_idx_decoded = time_idx

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames_decoded is None or None in frames_decoded:
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                if (
                    self.mode not in ["test"]
                    and (i_try % (self._num_retries // 8)) == 0
                ):
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            num_aug = (
                self.cfg.DATA.TRAIN_CROP_NUM_SPATIAL * self.cfg.AUG.NUM_SAMPLE
                if self.mode in ["train"]
                else 1
            )
            num_out = num_aug * num_decode
            f_out, time_idx_out = [None] * num_out, [None] * num_out
            idx = -1
            label = self._labels[index]

            for i in range(num_decode):
                for _ in range(num_aug):
                    idx += 1
                    f_out[idx] = frames_decoded[i].clone()
                    #time_idx_out[idx] = time_idx_decoded[i, :]

                    f_out[idx] = f_out[idx].float()
                    f_out[idx] = f_out[idx] / 255.0
                    f_out[idx] = f_out[idx].permute(3, 0, 1, 2)

                    scl, asp = (
                        self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
                        self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
                    )
                    relative_scales = (
                        None
                        if (self.mode not in ["train"] or len(scl) == 0)
                        else scl
                    )
                    relative_aspect = (
                        None
                        if (self.mode not in ["train"] or len(asp) == 0)
                        else asp
                    )
                    
                    f_out[idx] = utils.spatial_sampling(
                        f_out[idx],
                        spatial_idx=spatial_sample_index,
                        min_scale=min_scale[i],
                        max_scale=max_scale[i],
                        crop_size=crop_size[i],
                        random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                        inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                        aspect_ratio=relative_aspect,
                        scale=relative_scales,
                        motion_shift=self.cfg.DATA.TRAIN_JITTER_MOTION_SHIFT
                        if self.mode in ["train"]
                        else False,
                    )

                    f_out[idx] = utils.pack_pathway_output(self.cfg, f_out[idx])
            frames = f_out[0] if num_out == 1 else f_out
            time_idx = np.array(time_idx_out)
            if num_aug > 1:
                label = [label] * num_aug
                index = [index] * num_aug
            #print(num_aug)
            if self.mode == 'train':
                frames_1 = frames[0][0]
            else:
                frames_1 = frames[0]
            C,T,H,W = frames_1.shape
            if T > 16:
                frames_1 = frames_1[:,:16,...]
            else:
                frames_1 = torch.cat([frames_1, torch.zeros(C, T-16, H, W)], dim=1)
            if self.mode == 'train':
                frames_2 = frames[1][0]
            else:
                frames_2 = frames_1
            C,T,H,W = frames_2.shape
            if T > 16:
                frames_2 = frames_2[:,:16,...]
            else:
                frames_2 = torch.cat([frames_2, torch.zeros(C, 16-T, H, W)], dim=1)
            #print(label[0].shape)
            index = torch.zeros(1)
            time_idx = torch.zeros(1)
            diff_frames_1 = torch.zeros_like(frames_1)
            diff_frames_2 = torch.zeros_like(frames_2)
            diff_frames_1[:,1:,...] = torch.abs(frames_2[:,1:,...] - frames_2[:,:-1,...])
            frames_1 = torch.cat([frames_1, diff_frames_1], dim=0)# 50% percentage of MCTKD, introduced in the supplementary materials
            frames_2 = torch.cat([frames_2, diff_frames_2], dim=0)           
            if self.mode == 'train':            
                return [[frames_1],[frames_2]],torch.Tensor(self.embeddings).unsqueeze(0), label, index, time_idx, {}
            else:
                return [[frames_1]],torch.Tensor(self.embeddings).unsqueeze(0), label, index, time_idx, {}    
        else:
            raise RuntimeError(
                "Failed to fetch video idx {} from {}; after {} trials".format(
                    index, self._path_to_videos[index], i_try
                )
            )

    def _frame_to_list_img(self, frames):
        img_list = [
            transforms.ToPILImage()(frames[i]) for i in range(frames.size(0))
        ]
        return img_list

    def _list_img_to_frames(self, img_list):
        img_list = [transforms.ToTensor()(img) for img in img_list]
        return torch.stack(img_list)

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
        return len(self._path_to_videos)
