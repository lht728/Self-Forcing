from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

        # v2v ODE 初始化: 若 lmdb 内额外存了源条件 latent(由 Bernini teacher 离线生成), 则一并读取
        self.cond_latent_shape = None
        with self.env.begin() as txn:
            if txn.get(b"cond_latent_shape") is not None:
                self.cond_latent_shape = get_array_shape_from_lmdb(self.env, 'cond_latent')

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        out = {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }
        if self.cond_latent_shape is not None:
            cond_latent = retrieve_row_from_lmdb(
                self.env,
                "cond_latent", np.float16, idx, shape=self.cond_latent_shape[1:]
            )
            out["cond_latent"] = torch.tensor(cond_latent, dtype=torch.float32)
        return out


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }


class V2VVideoDataset(Dataset):
    """ReCo-Data 风格的 v2v 蒸馏数据集。

    只取源视频(src_video) + 编辑指令(prompt) 作为 v2v 条件与提示词，
    丢弃 tar_video（DMD 蒸馏 data-free，真分布来自 Bernini teacher）。

    每个样本输出:
        - prompts:   str
        - src_video: Tensor [C, F, H, W]，范围 [-1, 1]，F 为像素帧数(默认 81)
    """

    def __init__(
        self,
        data_path: str,
        base_video_folder: str,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        prompt_key: str = "instruction_final_refine",
        src_key: str = "src_video",
        target_fps: int = 16,
        max_pair: int = int(1e8),
    ):
        self.base_video_folder = base_video_folder
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.prompt_key = prompt_key
        self.src_key = src_key
        self.target_fps = target_fps

        # data_path 支持单个 *.json 或包含多个 *_data_configs.json 的目录(混合任务)
        json_files = []
        if os.path.isdir(data_path):
            for f in sorted(os.listdir(data_path)):
                if f.endswith("_data_configs.json"):
                    json_files.append(os.path.join(data_path, f))
            if not json_files:
                # 兼容 ReCo-Data 子任务目录(add/remove/replace/style)各含一个 json
                for sub in sorted(os.listdir(data_path)):
                    sub_dir = os.path.join(data_path, sub)
                    if not os.path.isdir(sub_dir):
                        continue
                    for f in sorted(os.listdir(sub_dir)):
                        if f.endswith("_data_configs.json"):
                            json_files.append(os.path.join(sub_dir, f))
        else:
            json_files = [data_path]

        self.items = []
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                if not isinstance(d, dict):
                    continue
                if d.get(self.src_key) and d.get(self.prompt_key):
                    self.items.append(d)

        self.items = self.items[:max_pair]
        if len(self.items) == 0:
            raise RuntimeError(
                f"V2VVideoDataset: 在 {data_path} 未找到有效样本"
                f"(需要字段 '{self.src_key}' 与 '{self.prompt_key}')"
            )

    def __len__(self):
        return len(self.items)

    def _sample_indices(self, total_frames, video_fps):
        # 按 target_fps 重采样到 num_frames（不足则重复末帧，超过则取前 num_frames）
        step = max(video_fps / float(self.target_fps), 1e-6)
        indices = np.arange(0, total_frames, step).astype(int)
        if len(indices) < self.num_frames:
            pad = np.array([indices[-1]] * (self.num_frames - len(indices)))
            indices = np.concatenate([indices, pad])
        indices = indices[: self.num_frames]
        return np.clip(indices, 0, total_frames - 1)

    def _load_video(self, path):
        import decord

        vr = decord.VideoReader(path, width=self.width, height=self.height)
        indices = self._sample_indices(len(vr), vr.get_avg_fps())
        frames = vr.get_batch(list(indices)).asnumpy()  # [F, H, W, C] uint8
        video = torch.from_numpy(frames).float().div_(255.0).mul_(2.0).sub_(1.0)
        return video.permute(3, 0, 1, 2).contiguous()  # [C, F, H, W]

    def __getitem__(self, idx):
        # 读视频失败时顺延到下一个样本，避免污染训练
        for offset in range(len(self.items)):
            item = self.items[(idx + offset) % len(self.items)]
            src_path = os.path.join(self.base_video_folder, item[self.src_key])
            try:
                src_video = self._load_video(src_path)
            except Exception as e:  # noqa: BLE001
                if offset == 0:
                    print(f"[V2VVideoDataset] 读取失败 {src_path}: {e}，顺延样本")
                continue
            return {
                "prompts": item[self.prompt_key],
                "src_video": src_video,
                "idx": idx,
            }
        raise RuntimeError("V2VVideoDataset: 连续读取视频失败")


class V2VPairedVideoDataset(V2VVideoDataset):
    """teacher 短训用的配对数据集: 同时返回源视频 + 目标(编辑后)视频 + 指令。

    继承 V2VVideoDataset 的解析/读帧逻辑, 额外读取 tar_video 作为监督目标。
    输出:
        - prompts:   str
        - src_video: Tensor [C, F, H, W] in [-1, 1] (条件)
        - tar_video: Tensor [C, F, H, W] in [-1, 1] (监督目标)
    """

    def __init__(self, *args, tar_key: str = "tar_video", **kwargs):
        self.tar_key = tar_key
        super().__init__(*args, **kwargs)
        self.items = [d for d in self.items if d.get(self.tar_key)]
        if len(self.items) == 0:
            raise RuntimeError(
                f"V2VPairedVideoDataset: 未找到含 '{self.tar_key}' 的样本")

    def __getitem__(self, idx):
        for offset in range(len(self.items)):
            item = self.items[(idx + offset) % len(self.items)]
            src_path = os.path.join(self.base_video_folder, item[self.src_key])
            tar_path = os.path.join(self.base_video_folder, item[self.tar_key])
            try:
                src_video = self._load_video(src_path)
                tar_video = self._load_video(tar_path)
            except Exception as e:  # noqa: BLE001
                if offset == 0:
                    print(f"[V2VPairedVideoDataset] 读取失败 {src_path}: {e}，顺延样本")
                continue
            return {
                "prompts": item[self.prompt_key],
                "src_video": src_video,
                "tar_video": tar_video,
                "idx": idx,
            }
        raise RuntimeError("V2VPairedVideoDataset: 连续读取视频失败")


def cycle(dl):
    while True:
        for data in dl:
            yield data
