from torch.utils import data
from easyvolcap.engine import DATASETS
from easyvolcap.utils.base_utils import dotdict
import numpy as np
import os


@DATASETS.register_module()
class DD100M(data.Dataset):
    def __init__(self, data_root, split='train', interval=None, dtype='pos3d', move=8):
        self.dances = {'rotmat':[], 'pos3d':[]}
        dtypes = ['rotmat', 'pos3d']
        self.dtype = dtype
        self.split = 'TRAIN' if split == 'train' else 'VAL'

        self.names = []

        for fname in os.listdir(os.path.join(data_root, 'pos3d', split)):
            pos_path = os.path.join(data_root, 'pos3d', split, fname)
            rot_path = os.path.join(data_root, 'rotmat', split, fname)

            pos3d = np.load(pos_path)
            rotmat = np.load(rot_path)[:, 3:]

            root = pos3d[:, :3]
            pos3d = pos3d - np.tile(root, (1, 55))
            pos3d[:, :3] = root

            left_twist = pos3d[:, 60:63]
            pos3d[:, 75:120] = (pos3d[:, 75:120] - np.tile(left_twist, (1, 15))) * 10

            right_twist = pos3d[:, 63:66]
            pos3d[:, 120:165] = (pos3d[:, 120:165] - np.tile(right_twist, (1, 15))) * 10

            assert pos3d.shape[0] == rotmat.shape[0], f"Length mismatch in {fname}"

            if interval is not None and interval != 'None':
                seq_len = pos3d.shape[0]
                for i in range(0, seq_len, move):
                    pos3d_sub = pos3d[i: i + interval]
                    rotmat_sub = rotmat[i: i + interval]
                    if len(pos3d_sub) != interval:
                        continue
                    self.dances['pos3d'].append(pos3d_sub)
                    self.dances['rotmat'].append(rotmat_sub)
                    self.names.append(fname)
            else:
                min_len = pos3d.shape[0] // move * move
                self.dances['pos3d'].append(pos3d[:min_len])
                self.dances['rotmat'].append(rotmat[:min_len])
                self.names.append(fname)

    def __len__(self):
        return len(self.dances['pos3d'])

    def __getitem__(self, index):
        meta = dotdict(file_name=self.names[index])
        output = dotdict(meta=meta,
                         pos3d=self.dances['pos3d'][index],
                         rotmat=self.dances['rotmat'][index])
        return output


@DATASETS.register_module()
class DD100lfAll(data.Dataset):
    def __init__(self, music_root, motion_root, split='train', interval=None, dtype='pos3d', move=8, music_dance_rate=1,
                 expansion=1, contact_root=None, drop_prob=0.0):
        self.dances = {'rotmatl': [], 'rotmatf': [], 'pos3dl': [], 'pos3df': [], 'music': [], 'contact': []}
        self.dtype = dtype
        self.expansion = expansion
        self.names = []
        self.split = 'TRAIN' if split == 'train' else 'VAL'
        self.drop_prob = drop_prob

        music_files = {}
        agent_files = {'leader': {}, 'follower': {}}

        contact_files = {}
        if contact_root is not None:
            for mname in os.listdir(os.path.join(contact_root, split)):
                path = os.path.join(contact_root, split, mname)
                contact_files[mname[:-4]] = path

        for mname in os.listdir(os.path.join(music_root, 'feature', split)):
            path = os.path.join(music_root, 'feature', split, mname)
            music_files[mname[:-4]] = path

        for fname in os.listdir(os.path.join(motion_root, 'pos3d', split)):
            path = os.path.join(motion_root, 'pos3d', split, fname)
            if path.endswith('_00.npy'):
                agent_files['follower'][fname[:-7]] = path
            elif path.endswith('_01.npy'):
                agent_files['leader'][fname[:-7]] = path

        for take in agent_files['follower']:
            if take not in agent_files['leader'] or take not in music_files:
                continue

            leader_pos = np.load(agent_files['leader'][take])
            follower_pos = np.load(agent_files['follower'][take])
            leader_rot = np.load(agent_files['leader'][take].replace('pos3d', 'rotmat'))[:, 3:]
            follower_rot = np.load(agent_files['follower'][take].replace('pos3d', 'rotmat'))[:, 3:]

            def preprocess_pos(pos):
                root = pos[:, :3]
                pos = pos - np.tile(root, (1, 55))
                pos[:, :3] = root
                pos[:, 75:120] = (pos[:, 75:120] - np.tile(pos[:, 60:63], (1, 15))) * 10
                pos[:, 120:165] = (pos[:, 120:165] - np.tile(pos[:, 63:66], (1, 15))) * 10
                return pos

            leader_pos = preprocess_pos(leader_pos).astype(np.float32)
            follower_pos = preprocess_pos(follower_pos).astype(np.float32)

            music = np.load(music_files[take]).astype(np.float32)
            contact = np.load(contact_files[take]).astype(np.float32) if contact_root else None

            seq_len = min(len(leader_pos), len(follower_pos), len(music), len(contact)) if contact is not None \
                else min(len(leader_pos), len(follower_pos), len(music))

            if interval is not None and interval != 'None':
                for i in range(0, seq_len, move):
                    lpos = leader_pos[i:i + interval]
                    fpos = follower_pos[i:i + interval]
                    lrot = leader_rot[i:i + interval]
                    frot = follower_rot[i:i + interval]
                    mus = music[i // music_dance_rate: i // music_dance_rate + interval // music_dance_rate]
                    con = contact[i:i + interval] if contact is not None else 0.0

                    if len(lpos) != interval or len(fpos) != interval or len(mus) != interval // music_dance_rate or \
                            (contact is not None and len(con) != interval):
                        continue

                    if contact_root is not None and self.drop_prob > 0.0:
                        if np.all(con == 0.0) and np.random.rand() < self.drop_prob:
                            continue

                    self.dances['pos3dl'].append(lpos)
                    self.dances['pos3df'].append(fpos)
                    self.dances['rotmatl'].append(lrot)
                    self.dances['rotmatf'].append(frot)
                    self.dances['music'].append(mus)
                    self.dances['contact'].append(con)
                    self.names.append(take)
            else:
                valid_len = seq_len // move * move
                self.dances['pos3dl'].append(leader_pos[:valid_len])
                self.dances['pos3df'].append(follower_pos[:valid_len])
                self.dances['rotmatl'].append(leader_rot[:valid_len])
                self.dances['rotmatf'].append(follower_rot[:valid_len])
                self.dances['music'].append(music[:valid_len])
                self.dances['contact'].append(contact[:valid_len] if contact is not None else 0.0)
                self.names.append(take)

    def __len__(self):
        return len(self.dances['pos3dl']) * self.expansion

    def __getitem__(self, index):
        index = index // self.expansion
        meta = dotdict(file_name=self.names[index])
        output = dotdict(meta=meta,
                         pos3dl=self.dances['pos3dl'][index],
                         pos3df=self.dances['pos3df'][index],
                         rotmatl=self.dances['rotmatl'][index],
                         rotmatf=self.dances['rotmatf'][index],
                         music=self.dances['music'][index],
                         contact=self.dances['contact'][index])
        return output


