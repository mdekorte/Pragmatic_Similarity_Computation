import csv
import torch
import os
import feature_extractor as fe
import feature_selection as fs
from similarity_metrics import get_cosine_similarity, get_euclidean_distance
from pathlib import Path
import time
import pygame
pygame.mixer.init()

class SimilarityFinder:
    def __init__(self, feature_selection=False, directory_path="", ssl_model='hubert_l', clips_for_comparison_path='data/dral_en.csv'):
        self.directory_path = Path(directory_path).as_posix() + "/"
        self.feature_selection = feature_selection
        self.clips_for_comparison = self.read_clips(clips_for_comparison_path)
        self.language = 'spanish' if '_es.csv' in clips_for_comparison_path else 'english'
        self.times = []
        assert ssl_model in ['hubert_l', 'wav2vec_l', 'wavlm_l'], "ssl_model must be 'hubert_l', 'wav2vec_l', or 'wavlm_l'"
        self.feature_extractor = fe.FeatureExtractor(ssl_model)

    # Reads a CSV file and returns a dictionary of clip paths and their feature averages
    # Assumes the following CSV format:
    # Row 1 + 3n: file path
    # Row 2 + 3n: feature averages (comma-separated)
    # Row 3 + 3n: empty row'
    def read_clips(self, path):
        full_path = os.path.join(self.directory_path, path)
        clip_dic = {}
        with open(full_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            counter = 1
            for row in csv_reader:
                if counter == 1:
                    file_path = row[0]
                if counter == 2:
                    features_avg = [float(x) for x in row]
                    clip_dic[file_path] = features_avg
                if counter == 3:
                    counter = 0
                counter += 1
        return clip_dic


    def find_similar(self, clip_to_find, metric='cosine'):
        similarities = []

        clip_to_find_avg = self.feature_extractor.get_24th_layer_features_averages(clip_to_find)
        clip_to_find_avg = fs.remove_losing_features(clip_to_find_avg, self.language)

        dataset_to_search = self.clips_for_comparison

        for test_clip in dataset_to_search:
            file_dir = os.path.dirname(os.path.abspath(__file__))
            test_clip_path = os.path.abspath(os.path.join(file_dir, test_clip))

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

    def get_average_times(self):
        print(f"average_times={sum(self.times)/len(self.times)}")