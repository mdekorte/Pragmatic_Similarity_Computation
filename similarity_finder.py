import csv
import torch
import os
import feature_extractor as fe
import feature_selection as fs
from pathlib import Path
import time
import pygame
pygame.mixer.init()

class SimilarityFinder:
    def __init__(self, feature_selection=False, directory_path="", clips_for_comparison_path='data/dral_en.csv'):
        self.directory_path = Path(directory_path).as_posix() + "/"
        self.feature_selection = feature_selection
        self.clips_for_comparison = self.read_clips(clips_for_comparison_path)
        self.language = 'spanish' if '_es.csv' in clips_for_comparison_path else 'english'
        self.times = []
        self.feature_extractor = fe.FeatureExtractor('hubert_l')

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


    def find_similar(self, clip_to_find):
        cos_similarities = []
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
            cos_sim = torch.nn.functional.cosine_similarity(torch.tensor(clip_to_find_avg), torch.tensor(test_clip_avg), dim=0)
            cos_similarities.append((cos_sim, test_clip_path))
            # ed = math.dist(clip_to_find_avg, test_clip_avg)
            #euclidean_distances.append((ed, test_clip))
        cos_similarities_sorted = sorted(cos_similarities, key=lambda tup: tup[0], reverse=True)
        return cos_similarities_sorted[0], \
            cos_similarities_sorted[1], \
            cos_similarities_sorted[2], \
            cos_similarities_sorted[999], \
            cos_similarities_sorted[1499], \
            cos_similarities_sorted[-1]

    def play_clip(self, file_name):

        pygame.mixer.music.load(file_name)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.5)
        time.sleep(1)

    def get_average_times(self):
        print(f"average_times={sum(self.times)/len(self.times)}")