import os
import numpy as np
import feature_extractor as fe
import feature_selection as fs
from similarity_metrics import get_cosine_similarity, get_euclidean_distance
from pathlib import Path
import time
import pygame
pygame.mixer.init()

class SimilarityFinder:
    def __init__(self, feature_selection=False, directory_path="", ssl_model='hubert_l', filelist='data/filelists/train_filelist_en.txt'):
        self.directory_path = Path(directory_path).as_posix() + "/"
        self.feature_selection = feature_selection
        self.language = 'spanish' if '_es.txt' in filelist else 'english'
        self.clips_for_comparison = self.get_features(os.path.join(self.directory_path, filelist), os.path.join(self.directory_path, 'data', 'features'))
        self.times = []
        assert ssl_model in ['hubert_l', 'wav2vec_l', 'wavlm_l'], "ssl_model must be 'hubert_l', 'wav2vec_l', or 'wavlm_l'"
        self.feature_extractor = fe.FeatureExtractor(ssl_model)


    def get_features(self, filelist_loc, feature_dir):
        with open(filelist_loc, 'r') as f:
            filelist_items = [line.strip() for line in f if line.strip()]
        clip_dict = {}
        for filename in filelist_items:
            feature_loc = os.path.join(feature_dir, filename.replace('.wav', '_features.npy'))
            features = np.load(feature_loc)
            if self.language == 'spanish':
                features = fs.extract_winners(features, 'spanish') if self.feature_selection else features
            elif self.language == 'english':
                features = fs.extract_winners(features, 'english') if self.feature_selection else features
            else:
                raise ValueError("Language must be 'spanish' or 'english'")
            clip_dict[filename] = features
        return clip_dict

    def find_similar(self, clip_to_find, target_loc, extract_with_padding=False, metric='cosine'):
        print(f"Finding similar clips to: {clip_to_find} using {metric} similarity metric.")
        similarities = []

        clip_to_find_avg = self.feature_extractor.get_24th_layer_features_averages(clip_to_find, extract_with_padding=extract_with_padding)
        clip_to_find_avg = fs.extract_winners(clip_to_find_avg, self.language)

        dataset_to_search = self.clips_for_comparison

        for test_clip in dataset_to_search:
            test_clip_path = os.path.abspath(os.path.join(target_loc, test_clip))

            clip_to_find = os.path.abspath(clip_to_find)
            if os.path.normcase(test_clip_path) == os.path.normcase(clip_to_find):
                continue

            test_clip_avg = dataset_to_search[test_clip]
            if metric == 'cosine':
                # use cosine similarity to compute pragmatic likeness
                similarities.append((get_cosine_similarity(clip_to_find_avg, test_clip_avg), test_clip_path))
            elif metric == 'euclidean':
                # alternatively, use euclidean distance
                similarities.append((get_euclidean_distance(clip_to_find_avg, test_clip_avg), test_clip_path))
            else:
                raise ValueError("Invalid metric. Use 'cosine' or 'euclidean'.")
        if metric == 'cosine':
            # sort descending (higher is more similar)
            similarities_sorted = sorted(similarities, reverse=True)
        elif metric == 'euclidean':
            # sort ascending (lower is more similar)
            similarities_sorted = sorted(similarities)
        else:
            raise ValueError("Invalid metric. Use 'cosine' or 'euclidean'.")

        # return max, 2nd max, 3rd max, 1st quartile, median, 3rd quartile, min
        return similarities_sorted[0], \
            similarities_sorted[1], \
            similarities_sorted[2], \
            similarities_sorted[int(len(similarities_sorted)/4)], \
            similarities_sorted[int(len(similarities_sorted)/2)], \
            similarities_sorted[int(3*len(similarities_sorted)/4)], \
            similarities_sorted[-1]

    # play the audio clip, and allow some time in between clips so that listener can distinguish them
    def play_clip(self, file_name):
        pygame.mixer.music.load(file_name)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.5)
        time.sleep(1)
